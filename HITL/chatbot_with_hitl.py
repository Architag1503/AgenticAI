# backend.py

from langgraph.graph import StateGraph, START # StateGraph manages the execution flow, START is the initial node
from typing import TypedDict, Annotated # TypedDict defines state structure, Annotated allows custom reducers
from langchain_core.messages import BaseMessage, HumanMessage # Message types for LLM input/output
from langchain_openai import ChatOpenAI # OpenAI LLM wrapper
from langgraph.checkpoint.memory import MemorySaver # In-memory checkpointing for persisting state (required for HITL)
from langgraph.graph.message import add_messages # Reducer function to append new messages to existing state
from langgraph.prebuilt import ToolNode, tools_condition # Pre-built node for executing tools, condition to route back or to tools
from langchain_core.tools import tool # Decorator to define custom tools
from langgraph.types import interrupt, Command # interrupt pauses the graph for HITL, Command allows resuming with specific state
from dotenv import load_dotenv
import requests

load_dotenv()

# -------------------
# 1. LLM
# -------------------
llm = ChatOpenAI()

# -------------------
# 2. Tools
# -------------------
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=4HEUC3GUJW74POW8"
    )
    r = requests.get(url)
    return r.json()


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    # HUMAN-IN-THE-LOOP (HITL) IMPLEMENTATION:
    # `interrupt` pauses the execution of the graph at this exact point.
    # It returns control to the caller (user/frontend), surfacing the prompt string.
    # The graph state is saved in the checkpointer (`MemorySaver`).
    # Execution will only resume when the caller invokes the graph again using `Command(resume=...)`.
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }


tools = [get_stock_price, purchase_stock]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    # `messages` uses the `add_messages` reducer. 
    # This means when a node returns a new message, it is appended to the list, not overwritten.
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer (in-memory)
# -------------------
memory = MemorySaver()

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

# `tools_condition` checks if the LLM returned any tool_calls. 
# If yes, routes to "tools". If no, routes to END.
graph.add_conditional_edges("chat_node", tools_condition)

# After tool execution, always return to the chat node so the LLM can see the tool output
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=memory)

# -------------------
# 7. Simple usage example (CLI with HITL)
# -------------------
if __name__ == "__main__":
    
    # Use a fixed thread_id so the conversation is persisted in memory
    thread_id = "demo-thread"

    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Build initial state for this turn
        state = {"messages": [HumanMessage(content=user_input)]}

        # Run the graph (may hit an interrupt)
        result = chatbot.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )

        # Check for HITL interrupt from purchase_stock
        interrupts = result.get("__interrupt__", [])

        if interrupts:
            # Our interrupt payload is the string we passed to interrupt(...)
            prompt_to_human = interrupts[0].value
            print(f"HITL: {prompt_to_human}")
            decision = input("Your decision: ").strip().lower()

            # Resume graph with the human decision ("yes" / "no" / whatever)
            # `Command(resume=decision)` feeds the user's response back into the `interrupt` call in the tool.
            # The graph picks up execution exactly where it paused.
            result = chatbot.invoke(
                Command(resume=decision),
                config={"configurable": {"thread_id": thread_id}},
            )

        # Get the latest message from the assistant
        messages = result["messages"]
        last_msg = messages[-1]
        print(f"Bot: {last_msg.content}\n")