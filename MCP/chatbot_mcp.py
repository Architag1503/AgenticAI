from langgraph.graph import StateGraph , START , END
from typing import TypedDict , Annotated
from langchain_core.messages import HumanMessage , BaseMessage
from langchain_mistralai import ChatMistralAI
from langgraph.graph.message import add_messages

from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode , tools_condition
from langchain_core.tools import tool
import sys
from rich import print
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

llm = ChatMistralAI(model="mistral-small-2506")

client = MultiServerMCPClient(
    {
        "arith" : {
            "transport" : "stdio",
            "command" : sys.executable,
            "args": [r"A:\Agentic AI\MCP\main.py"]
        },
        "expense_tracker" : {
            "transport" : "stdio",
            "command" : sys.executable,
            "args": [r"A:\Agentic AI\MCP\main2.py"]
        },

    }
)


# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage] , add_messages]

async def build_graph():

    tools = await client.get_tools()
    print(tools)

    # Make the LLM tool-aware
    llm_with_tool = llm.bind_tools(tools , tool_choice="auto")

    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await llm_with_tool.ainvoke(messages)
        return {"messages" : [response]}
    
    tool_node = ToolNode(tools) #Execute tools calls

    # graph structure
    graph = StateGraph(ChatState)
    graph.add_node("chat_node" , chat_node)
    graph.add_node("tools" , tool_node)
    
    graph.add_edge(START , "chat_node")
    
    graph.add_conditional_edges("chat_node" , tools_condition)
    
    # graph.add_edge("tools" , END) # this sometimes not solve the complex queries. So in order o resolve the we creating the loop between tool and chat_node
    graph.add_edge("tools", "chat_node")
    graph.add_edge("chat_node" , END)
    
    chatbot = graph.compile()
    return chatbot

async def main():
    chatbot = await build_graph()

    # Running the graph
    # result = await chatbot.ainvoke({"messages" : [HumanMessage(content="Find the modulus of 124589 and 23 and give answer like a cricke commentator")]})     
    # result = await chatbot.ainvoke({"messages" : [HumanMessage(content="Add a new expense for 150.50 under category 'Groceries' with description 'Supermarket' on 2026-07-31, and then list my expenses for July 2026.")]})     
    # result = await chatbot.ainvoke({"messages" : [HumanMessage(content="Add an expense of Rs 5000 for Udemy Course on 25th July 2026")]})     
    # result = await chatbot.ainvoke({"messages" : [HumanMessage(content="I want you to record the following expenses in my expense tracker. On 2026-07-01, I spent ₹250 on Food for breakfast at a café. On 2026-07-02, I spent ₹1,200 on Groceries for weekly supermarket shopping. On 2026-07-03, I spent ₹180 on Transport for a metro pass. On 2026-07-04, I spent ₹2,500 on Shopping to buy a pair of shoes. On 2026-07-05, I spent ₹650 on Entertainment for a movie and snacks. On 2026-07-06, I paid ₹850 for Utilities as my electricity bill. On 2026-07-07, I spent ₹320 on Food for dinner at a restaurant. On 2026-07-08, I spent ₹1,500 on Healthcare for medicines and a doctor's consultation. On 2026-07-09, I spent ₹450 on Transport for fuel. On 2026-07-10, I spent ₹3,200 on Electronics to purchase a wireless keyboard. Please add each of these as separate expense entries with the appropriate amount, category, date, and description.")]})     
    result = await chatbot.ainvoke({"messages" : [HumanMessage(content="Give the list of all expenses of date 2026-07-01")]})     

    print(result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())
