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

    

    # st.session_state['message_history'].append({'role' : 'assistant' , 'content' : ai_messages})
    with st.chat_message('assistant'):

        ai_message = st.write_stream(
            message_chunk.content for message_chunk , metadata in chatbot.stream(
                {'messages' : [HumanMessage(content=user_input)]},
                config={"configurable" : {"thread_id" : "thread-1"}},
                stream_mode='messages'
            )
        )

    st.session_state['message_history'].append({'role' : 'assistant' , 'content' : ai_message})
    
        