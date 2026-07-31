from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import aiosqlite
import requests
import asyncio
import threading
import sys

load_dotenv()

# ==========================================
# WHY: Dedicated Async Loop for Backend Tasks
# Streamlit runs synchronously on its own main thread. LangGraph and MCP clients heavily 
# rely on async/await. To bridge the gap safely without "event loop already running" errors, 
# we create a dedicated background thread that constantly runs an asyncio event loop.
# ==========================================
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    """Internal helper to schedule a coroutine on our background event loop."""
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

def run_async(coro):
    """
    Synchronously wait for an async task to finish.
    WHY: This allows Streamlit's synchronous code to call async backend setup functions 
    (like loading tools or checkpointers) and wait for the results.
    """
    return _submit_async(coro).result()

def submit_async_task(coro):
    """
    Schedule a coroutine without blocking.
    WHY: Used when we want to start streaming the LLM's response in the background 
    while the Streamlit UI immediately starts reading from a queue.
    """
    return _submit_async(coro)

# -------------------
# 1. LLM
# -------------------
llm = ChatMistralAI()

# -------------------
# 2. Tools Configuration
# -------------------
# Native LangChain Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

# MCP Client Setup
# WHY: Here we connect to external MCP servers (arith and expense_tracker). 
# This separates tool logic into external processes, making the architecture highly extensible.
client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            # WHY sys.executable: Ensures we use the current virtual environment's Python 
            # to spawn the server subprocess, preventing missing dependency errors.
            "command": sys.executable,
            "args": [r"A:\Agentic AI\MCP\main.py"],
        },
        "expense_tracker": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [r"A:\Agentic AI\MCP\main2.py"]
        },
    }
)

def load_mcp_tools() -> list[BaseTool]:
    """
    Fetch tools from the MCP servers.
    WHY try/except: If an MCP server crashes or is unavailable, this prevents the entire 
    Streamlit app from crashing. It simply falls back to loading zero MCP tools.
    """
    try:
        return run_async(client.get_tools())
    except Exception:
        return []

mcp_tools = load_mcp_tools()

# Combine native tools with MCP tools and bind them to the LLM
tools = [search_tool, get_stock_price, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    # WHY: Holds the conversation history. add_messages appends instead of overwriting.
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: ChatState):
    """LLM node that processes the conversation history and generates a response or tool request."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

# Tool node executes the requested tools. If no tools exist, it's None.
tool_node = ToolNode(tools) if tools else None

# -------------------
# 5. Checkpointer (Memory)
# -------------------
async def _init_checkpointer():
    # WHY SqliteSaver: A checkpointer saves the state of the graph at every step to a SQLite database. 
    # This provides persistent memory across different chat sessions (Thread IDs).
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)

checkpointer = run_async(_init_checkpointer())

# -------------------
# 6. Graph Assembly
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    # Route to tools if the LLM made a tool call, else route to END
    graph.add_conditional_edges("chat_node", tools_condition)
    # Loop back to the LLM after tool execution
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

# WHY compile with checkpointer: Compiling finalizes the graph. Passing the checkpointer 
# enables built-in long-term memory keyed by "thread_id".
chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Thread Helper Functions
# -------------------
async def _alist_threads():
    # WHY: Retrieves all unique thread IDs from the database so the user can select 
    # and resume past conversations in the UI sidebar.
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

def retrieve_all_threads():
    return run_async(_alist_threads())