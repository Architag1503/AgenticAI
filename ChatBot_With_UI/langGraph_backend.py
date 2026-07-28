from langgraph.graph import StateGraph , START , END
from langchain_core.messages import BaseMessage , HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict , Annotated
from langgraph.graph.message import add_messages
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv()

model = ChatMistralAI(model="mistral-small-2506")

class ChatState(TypedDict):

    messages : Annotated[list[BaseMessage], add_messages]

def chat_node(state : ChatState):
    messages =  state['messages']
    response = model.invoke(messages)

    return {"messages" : response}

graph = StateGraph(ChatState)
checkpointer = InMemorySaver()

# add nodes
graph.add_node("chat_node" , chat_node)

# add edges
graph.add_edge(START , "chat_node")
graph.add_edge("chat_node" , END)

# Compile the Graph
chatbot = graph.compile(checkpointer=checkpointer)

