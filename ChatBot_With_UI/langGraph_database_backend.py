from langgraph.graph import StateGraph , START , END
from langchain_core.messages import BaseMessage , HumanMessage
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict , Annotated
from langgraph.graph.message import add_messages
from langchain_mistralai import ChatMistralAI
import sqlite3
from dotenv import load_dotenv
load_dotenv()

model = ChatMistralAI(model="mistral-small-2506")

class ChatState(TypedDict):

    messages : Annotated[list[BaseMessage], add_messages]

def chat_node(state : ChatState):
    messages =  state['messages']
    response = model.invoke(messages)

    return {"messages" : response}


conn = sqlite3.connect(database='chatbot.db' , check_same_thread=False)

# checkpointer
checkpointer = SqliteSaver(conn=conn)


graph = StateGraph(ChatState)

# add nodes
graph.add_node("chat_node" , chat_node)

# add edges
graph.add_edge(START , "chat_node")
graph.add_edge("chat_node" , END)

# Compile the Graph
chatbot = graph.compile(checkpointer=checkpointer)

def retreive_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)