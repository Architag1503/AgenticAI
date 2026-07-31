from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_mistralai import ChatMistralAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import sys
from rich import print
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

# WHY load_dotenv: We need to load environment variables (like API keys) from a .env file 
# so the ChatMistralAI class can authenticate its API requests.
load_dotenv()

# WHY MistralAI: This initializes our LLM engine. We are using Mistral's small model 
# which is fast and supports function calling (tool use).
llm = ChatMistralAI(model="mistral-small-2506")

# WHY MultiServerMCPClient: This client connects to multiple MCP servers simultaneously.
# By passing a dictionary, we tell it how to spawn our Python scripts as subprocesses 
# and communicate with them via standard input/output (stdio).
client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            # WHY sys.executable: This ensures the subprocess uses the exact same Python 
            # virtual environment that this script is currently running in. If we used "python", 
            # it might accidentally use the system's global Python which lacks our installed packages.
            "command": sys.executable,
            "args": [r"A:\Agentic AI\MCP\main.py"]
        },
        "expense_tracker": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [r"A:\Agentic AI\MCP\main2.py"]
        }
    }
)

# State Definition
# WHY TypedDict: LangGraph requires a typed state dictionary to keep track of the conversation 
# as it loops through different nodes.
class ChatState(TypedDict):
    # WHY add_messages: This reducer function appends new messages to the existing list 
    # rather than overwriting the list completely. This preserves chat history.
    messages: Annotated[list[BaseMessage], add_messages]

async def build_graph():
    # WHY get_tools: This dynamically asks the MCP servers "What tools do you have available?"
    # It returns a list of LangChain-compatible tools derived directly from our MCP servers.
    tools = await client.get_tools()
    print("Loaded tools:", tools)

    # Make the LLM tool-aware
    # WHY bind_tools: We must explicitly give the LLM the schemas of our tools. 
    # This alters the LLM's system prompt behind the scenes so it knows how to request a tool call.
    llm_with_tool = llm.bind_tools(tools, tool_choice="auto")

    # Node 1: The LLM Chat Node
    # WHY: This node feeds the conversation history to the LLM. The LLM will either 
    # return a natural language response, OR it will return a "Tool Call" request.
    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await llm_with_tool.ainvoke(messages)
        return {"messages": [response]}
    
    # Node 2: The Tool Execution Node
    # WHY ToolNode: If the LLM requests a tool call, this node intercepts it, actually 
    # executes the Python function (via MCP), and returns the result as a ToolMessage.
    tool_node = ToolNode(tools) 

    # Graph Structure
    # WHY StateGraph: We are building a state machine where execution flows from node to node 
    # based on edges and conditions.
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)
    
    # Set the entry point of the graph
    graph.add_edge(START, "chat_node")
    
    # Conditional Edges
    # WHY tools_condition: This built-in LangGraph function checks the LLM's output. 
    # If the LLM wants to use a tool, it routes to the "tools" node. 
    # If the LLM just gave a normal text reply, it routes to END (finishing the loop).
    graph.add_conditional_edges("chat_node", tools_condition)
    
    # Loop back
    # WHY route tools -> chat_node: After a tool finishes executing, we MUST send the tool's 
    # result back to the LLM so the LLM can interpret the data and give a final answer.
    graph.add_edge("tools", "chat_node")
    graph.add_edge("chat_node", END)
    
    chatbot = graph.compile()
    return chatbot

async def main():
    chatbot = await build_graph()

    # Running the graph
    # WHY ainvoke: We asynchronously invoke the graph with an initial HumanMessage. 
    # The graph will loop between the LLM and the Tools until the LLM decides it has fully answered the prompt.
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="Add a new expense for 150.50 under category 'Groceries' with description 'Supermarket' on 2026-07-31, and then list my expenses for July 2026.")]})     

    # Print the final output from the LLM
    print(result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())
