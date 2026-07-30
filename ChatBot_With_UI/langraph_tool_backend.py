from langgraph.graph import StateGraph , START , END
from typing import TypedDict , Annotated
from langchain_core.messages import HumanMessage , BaseMessage
from langchain_mistralai import ChatMistralAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode , tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from rich import print

import requests
import sqlite3
import random

load_dotenv()

llm = ChatMistralAI(model="mistral-small-2506")

# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float , second_num: float , operation: str) -> dict:
    """Perform arithmetic calculations.

  Args:
      first_num: First number
      second_num: Second number
      operation: One of 'add', 'sub', 'mul', 'div'
  """

    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error" : "Division by zero not allowed"}
            result = first_num / second_num
        else:
            return {"error" : f"Unsupported operation : '{operation}'"}

        return {"first_num": first_num , "second_num": second_num, "operation": operation , "result" : result}
    except Exception as e:
        return {"error" : str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given (e.g 'AAPL' , 'TSLA')
    using Alpha Vantage with the API key in the URL
    """

    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=08CDBZKCNZ151UPG"

    r = requests.get(url)

    return r.json()


# Make the List of tools
tools = [get_stock_price , search_tool , calculator]

# Make the LLM tool-aware
llm_with_tool = llm.bind_tools(tools , tool_choice="auto")

# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage] , add_messages]


# graph nodes
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tool.invoke(messages)
    return {"messages" : [response]}

tool_node = ToolNode(tools) #Execute tools calls


# Checkpointer
conn = sqlite3.connect(database="chatbot.db" , check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# graph structure
graph = StateGraph(ChatState)
graph.add_node("chat_node" , chat_node)
graph.add_node("tools" , tool_node)

graph.add_edge(START , "chat_node")

graph.add_conditional_edges("chat_node" , tools_condition)

# graph.add_edge("tools" , END) # this sometimes not solve the complex queries. So in order o resolve the we creating the loop between tool and chat_node
graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node" , END)

chatbot = graph.compile(checkpointer=checkpointer)
chatbot

# Helper
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])

    return list(all_threads)



