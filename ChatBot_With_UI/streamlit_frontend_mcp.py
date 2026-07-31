import queue
import uuid

import streamlit as st
from langGraph_mcp_backend import chatbot, retrieve_all_threads, submit_async_task
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# =========================== Utilities ===========================
def generate_thread_id():
    """WHY UUID: Creates a globally unique identifier for every new chat session."""
    return uuid.uuid4()

def reset_chat():
    """WHY: Clears the current UI state and starts a fresh conversation thread."""
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    """Keeps track of all threads for the sidebar history list."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    """
    WHY: Queries the backend LangGraph checkpointer for a specific thread_id 
    to retrieve past messages. This is how chat history persistence works!
    """
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get("messages", [])


# ======================= Session Initialization ===================
# WHY st.session_state: Streamlit reruns the entire script from top to bottom on EVERY user interaction. 
# session_state is the only way to preserve variables (like message history) between reruns.
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("LangGraph MCP Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")
# Reverse the threads so the newest ones appear at the top
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        # WHY: When a user clicks a past thread, we update the current thread_id 
        # and reload the messages from the LangGraph database into the UI history.
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages

# ============================ Main UI ============================

# WHY Render history loop: Since Streamlit reruns on every input, we must redraw all 
# past messages on the screen every single time, reading from our session_state.
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # 1. Immediately show the user's message in the UI
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # WHY CONFIG: This config dictionary tells LangGraph exactly which thread 
    # to save this specific conversation step into.
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        # WHY a mutable holder: We need to update the Streamlit UI (st.status) dynamically 
        # when a tool starts and finishes, but we are inside a generator function. 
        # A dictionary allows us to modify the UI box reference safely.
        status_holder = {"box": None}

        def ai_only_stream():
            # WHY Queue: We use a thread-safe Queue to pass streaming data from the background 
            # async thread (where LangGraph runs) to the main Streamlit synchronous thread.
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    # WHY astream: Asynchronously streams the LangGraph execution steps. 
                    # We stream "messages" mode, so we get chunks as the LLM generates tokens 
                    # and as Tools return their outputs.
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put(None)

            # Kick off the async LangGraph process in the background thread
            submit_async_task(run_stream())

            # Read from the queue synchronously in Streamlit
            while True:
                item = event_queue.get()
                if item is None:
                    break
                message_chunk, metadata = item
                if message_chunk == "error":
                    raise metadata

                # WHY UI Status for Tools: If the graph is executing a ToolMessage, 
                # we show an expandable UI box so the user knows the AI is "thinking" or "fetching data"
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # WHY yield AIMessage: Streamlit's write_stream only expects the text tokens 
                # generated by the assistant. We filter out the internal ToolMessages.
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        # Execute the generator and type out the response in the UI
        ai_message = st.write_stream(ai_only_stream())

        # WHY finalize: Once the generation is completely done, we close the tool status box if one was opened.
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save the final generated assistant message into the session state so it persists on the next redraw.
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )