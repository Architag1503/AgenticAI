import streamlit as st
from langGraph_backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {"configurable" : {"thread_id" : "thread-1"}}

# st.session_state -> dictionary which is only refresh on refreshing the page not on hitting the enter
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


# load the conversation 
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type Here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role' : 'user' , 'content' : user_input})
    with st.chat_message('user'):
        st.text(user_input)

    response = chatbot.invoke({'messages' : [HumanMessage(content=user_input)]} , config=CONFIG)
    ai_messages = response['messages'][-1].content
    st.session_state['message_history'].append({'role' : 'assistant' , 'content' : ai_messages})
    with st.chat_message('assistant'):
        st.text(ai_messages)