"""
streamlit_HITL_frontend.py

A modern, ChatGPT/Notion-styled Streamlit frontend for the LangGraph
Human-in-the-Loop (HITL) Email Draft Assistant.

This file imports helper functions directly from `langGraph_HITL_backend.py`
and never modifies it. All workflow orchestration (LLM drafting, interrupt
handling, SMTP sending, SQLite persistence) lives in the backend; this file
is purely presentation and session-state management.
"""

from __future__ import annotations

from datetime import datetime, date
from typing import Any, Optional

import streamlit as st

from langGraph_HITL_backend import (
    start_email_workflow,
    resume_workflow,
    list_threads,
    load_thread,
    load_all_sent_emails,
    search_threads,
    delete_thread,
    EmailValidationError,
    SMTPSendError,
    DatabaseError,
    LLMGenerationError,
    WorkflowError,
)

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="AI Email Assistant",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Custom CSS — ChatGPT/Notion inspired, gradient accents, rounded cards
# --------------------------------------------------------------------------- #

def inject_custom_css() -> None:
    """Inject custom CSS for a modern, minimal, card-based UI."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Gradient hero header */
        .gradient-header {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
            padding: 2.2rem 2rem;
            border-radius: 20px;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.25);
        }
        .gradient-header h1 {
            color: #ffffff;
            font-weight: 800;
            font-size: 2.1rem;
            margin: 0;
        }
        .gradient-header p {
            color: rgba(255,255,255,0.9);
            font-size: 1rem;
            margin-top: 0.3rem;
            font-weight: 500;
        }

        /* Generic card */
        .app-card {
            background: var(--background-color, #ffffff);
            border: 1px solid rgba(120, 120, 120, 0.15);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            transition: box-shadow 0.2s ease, transform 0.2s ease;
        }
        .app-card:hover {
            box-shadow: 0 8px 24px rgba(0,0,0,0.10);
            transform: translateY(-1px);
        }

        .thread-card {
            background: rgba(120,120,120,0.06);
            border: 1px solid rgba(120,120,120,0.15);
            border-radius: 12px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.6rem;
            transition: background 0.2s ease;
        }
        .thread-card:hover {
            background: rgba(99,102,241,0.10);
        }

        .email-preview {
            background: rgba(120,120,120,0.05);
            border-left: 4px solid #6366f1;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            font-size: 0.95rem;
            line-height: 1.55;
            white-space: pre-wrap;
        }

        /* Status badges */
        .badge {
            display: inline-block;
            padding: 0.22rem 0.7rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .badge-pending { background: #fef3c7; color: #92400e; }
        .badge-approved { background: #d1fae5; color: #065f46; }
        .badge-rejected { background: #fee2e2; color: #991b1b; }
        .badge-edit { background: #dbeafe; color: #1e40af; }

        /* Buttons */
        .stButton>button {
            border-radius: 10px;
            font-weight: 600;
            border: none;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 14px rgba(0,0,0,0.12);
        }

        /* Rounded inputs */
        .stTextInput>div>div>input,
        .stTextArea textarea,
        .stSelectbox>div>div {
            border-radius: 10px !important;
        }

        /* Timeline */
        .timeline-item {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.35rem 0;
            font-size: 0.9rem;
        }
        .timeline-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #6366f1;
            flex-shrink: 0;
        }
        .timeline-line {
            width: 2px;
            height: 16px;
            background: rgba(120,120,120,0.3);
            margin-left: 4px;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(120,120,120,0.15);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Session state initialization
# --------------------------------------------------------------------------- #

def init_session_state() -> None:
    """Safely initialize all required session-state keys."""
    defaults: dict[str, Any] = {
        "thread_id": None,
        "draft": "",
        "subject": "",
        "sender": "",
        "receiver": "",
        "sender_password": "",
        "prompt": "",
        "workflow_state": "idle",   # idle | reviewing | sent | rejected
        "approval_status": "pending",  # pending | approved | rejected | edit
        "feedback": "",
        "history": [],
        "selected_thread": None,
        "search_query": "",
        "confirm_delete_id": None,
        "show_settings": False,
        "setup_step1_done": False,
        "setup_step2_done": False,
        "saved_sender_email": "",
        "saved_app_password": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_workflow_state() -> None:
    """Clear all workflow-related session state to start a fresh email."""
    for key in (
        "thread_id",
        "draft",
        "subject",
        "sender",
        "receiver",
        "sender_password",
        "prompt",
        "feedback",
    ):
        st.session_state[key] = ""
    st.session_state["thread_id"] = None
    st.session_state["workflow_state"] = "idle"
    st.session_state["approval_status"] = "pending"
    st.session_state["selected_thread"] = None


# --------------------------------------------------------------------------- #
# Small display helpers
# --------------------------------------------------------------------------- #

def status_badge(status: str) -> str:
    """Return HTML for a colored status badge."""
    status = (status or "pending").lower()
    mapping = {
        "approved": ("Approved", "badge-approved"),
        "pending": ("Pending", "badge-pending"),
        "rejected": ("Rejected", "badge-rejected"),
        "edit": ("Edited", "badge-edit"),
    }
    label, css_class = mapping.get(status, ("Pending", "badge-pending"))
    return f'<span class="badge {css_class}">{label}</span>'


def format_timestamp(raw: Optional[str]) -> str:
    """Format an ISO timestamp string into a friendly display string."""
    if not raw:
        return "—"
    try:
        dt = datetime.fromisoformat(raw)
        return dt.strftime("%b %d, %Y · %I:%M %p")
    except (ValueError, TypeError):
        return raw


def render_email_timeline(records: list[dict[str, Any]]) -> None:
    """Render a vertical timeline of a thread's revision/approval history."""
    st.markdown("**Email Timeline**")
    steps = []
    for idx, rec in enumerate(records):
        label = "Draft Generated" if idx == 0 else "Edited"
        steps.append(label)
        if rec.get("approved") == "approved":
            steps.append("Approved")
            steps.append("Sent")

    html_parts = []
    for i, step in enumerate(steps):
        html_parts.append(
            f'<div class="timeline-item"><div class="timeline-dot"></div>'
            f'<span>{step}</span></div>'
        )
        if i < len(steps) - 1:
            html_parts.append('<div class="timeline-line"></div>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Gmail Setup Wizard — guided, human-in-the-loop account setup
# --------------------------------------------------------------------------- #

def render_settings_wizard() -> None:
    """
    Render a guided setup wizard that helps the user enable 2-Step
    Verification and generate a Gmail App Password.

    Note: Google does not expose any API to enable 2-Step Verification or
    create an App Password on a user's behalf — both are security-sensitive
    actions that only exist behind Google's own login UI, and there is no
    supported way for a third-party app to perform them automatically.
    This wizard therefore uses a human-in-the-loop pattern: it deep-links
    the user to the exact Google settings page for each step, asks them to
    confirm once they've completed it, and then lets them paste the
    resulting app password back into the app so it doesn't need to be
    re-entered for every email.
    """
    st.markdown("### ⚙️ Gmail Account Setup")
    st.caption(
        "Google requires an App Password for third-party apps like this one — "
        "your normal account password will not work for SMTP. "
        "This wizard walks you through generating one. Each step is done on "
        "Google's own site; nothing here can change your Google account "
        "settings automatically."
    )

    with st.container():
        st.markdown('<div class="app-card">', unsafe_allow_html=True)

        # --- Step 1: Enable 2-Step Verification -------------------------- #
        st.markdown("**Step 1 — Enable 2-Step Verification**")
        st.write(
            "App Passwords only appear once 2-Step Verification is turned "
            "on for the Google account you'll send from."
        )
        st.link_button(
            "🔐 Open Google Security Settings",
            "https://myaccount.google.com/security",
            use_container_width=False,
        )
        step1_done = st.checkbox(
            "I've enabled 2-Step Verification on this Google account",
            value=st.session_state.get("setup_step1_done", False),
            key="setup_step1_checkbox",
        )
        st.session_state["setup_step1_done"] = step1_done

        st.markdown("---")

        # --- Step 2: Generate the App Password ---------------------------- #
        st.markdown("**Step 2 — Generate an App Password**")
        if not step1_done:
            st.info("Complete Step 1 first, then continue here.")
        else:
            st.write(
                "Open the App Passwords page, create one named e.g. "
                "\"AI Email Assistant\", and copy the 16-character code Google "
                "shows you."
            )
            st.link_button(
                "🔑 Open Google App Passwords",
                "https://myaccount.google.com/apppasswords",
                use_container_width=False,
            )

            with st.form("save_app_password_form", clear_on_submit=False):
                gmail_address = st.text_input(
                    "Gmail address this app password belongs to",
                    value=st.session_state.get("saved_sender_email", ""),
                    placeholder="you@gmail.com",
                )
                app_password = st.text_input(
                    "Paste the App Password Google gave you",
                    type="password",
                    placeholder="xxxx xxxx xxxx xxxx",
                )
                save_clicked = st.form_submit_button(
                    "💾 Save App Password", use_container_width=True, type="primary"
                )

            if save_clicked:
                cleaned_password = app_password.replace(" ", "").strip()
                if not gmail_address or not cleaned_password:
                    st.warning(
                        "Please enter both the Gmail address and the app password."
                    )
                elif len(cleaned_password) < 16:
                    st.warning(
                        "That doesn't look like a valid app password — it should "
                        "be 16 characters (spaces are fine, we'll strip them)."
                    )
                else:
                    st.session_state["saved_sender_email"] = gmail_address
                    st.session_state["saved_app_password"] = cleaned_password
                    st.session_state["setup_step2_done"] = True
                    st.success(
                        "App password saved for this session. It will now "
                        "pre-fill automatically when you compose a new email."
                    )

        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.get("saved_app_password"):
            st.success(
                f"✅ Ready to send as **{st.session_state.get('saved_sender_email')}**"
            )
            if st.button("Forget saved app password"):
                st.session_state["saved_app_password"] = ""
                st.session_state["saved_sender_email"] = ""
                st.session_state["setup_step2_done"] = False
                st.rerun()

    st.caption(
        "🔒 The app password is kept only in this browser session's memory — "
        "it is never written to disk or the database."
    )


# --------------------------------------------------------------------------- #
# Metrics row
# --------------------------------------------------------------------------- #

def render_metrics() -> None:
    """Render top-row metrics: total threads, sent emails, pending, today's."""
    try:
        threads = list_threads()
    except DatabaseError as exc:
        st.error(f"Could not load threads: {exc}")
        threads = []

    try:
        sent_emails = load_all_sent_emails()
    except DatabaseError as exc:
        st.error(f"Could not load sent emails: {exc}")
        sent_emails = []

    total_threads = len(threads)
    total_sent = len(sent_emails)

    pending_count = 0
    today_count = 0
    today_str = date.today().isoformat()
    for rec in sent_emails:
        sent_time = rec.get("sent_time") or ""
        if sent_time.startswith(today_str):
            today_count += 1

    # Pending = threads with no sent email yet
    sent_thread_ids = {rec.get("thread_id") for rec in sent_emails}
    pending_count = sum(1 for t in threads if t.get("thread_id") not in sent_thread_ids)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Threads", total_threads)
    col2.metric("Sent Emails", total_sent)
    col3.metric("Pending Reviews", pending_count)
    col4.metric("Today's Emails", today_count)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

def render_sidebar() -> None:
    """Render the sidebar: new email button, search, and thread history."""
    with st.sidebar:
        st.markdown("### 📧 AI Email Assistant")

        if st.button("➕ New Email", use_container_width=True, type="primary"):
            reset_workflow_state()
            st.session_state["show_settings"] = False
            st.rerun()

        if st.button("⚙️ Gmail Account Setup", use_container_width=True):
            st.session_state["show_settings"] = not st.session_state.get(
                "show_settings", False
            )
            st.rerun()

        st.divider()

        st.markdown("#### 🔍 Search Threads")
        query = st.text_input(
            "Search Thread",
            value=st.session_state.get("search_query", ""),
            placeholder="Search by subject, receiver, or content...",
            label_visibility="collapsed",
        )
        st.session_state["search_query"] = query

        if query:
            try:
                results = search_threads(query)
            except DatabaseError as exc:
                st.error(f"Search failed: {exc}")
                results = []

            if results:
                st.caption(f"{len(results)} result(s) found")
                for rec in results:
                    with st.container():
                        st.markdown(
                            f"""
                            <div class="thread-card">
                                <b>{rec.get('subject', '(no subject)')}</b><br/>
                                <span style="font-size:0.8rem;opacity:0.75;">
                                    To: {rec.get('receiver_email', '—')}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            "Open thread",
                            key=f"open_search_{rec.get('id')}",
                            use_container_width=True,
                        ):
                            st.session_state["selected_thread"] = rec.get("thread_id")
                            st.session_state["workflow_state"] = "viewing_history"
                            st.rerun()
            else:
                st.info("No matching threads found.")

        st.divider()

        st.markdown("#### 🗂️ Thread History")
        try:
            threads = list_threads()
        except DatabaseError as exc:
            st.error(f"Could not load threads: {exc}")
            threads = []

        if not threads:
            st.caption("No threads yet. Start a new email to begin.")

        for thread in threads:
            thread_id = thread.get("thread_id", "")
            subject = thread.get("subject") or "(no subject)"
            receiver = thread.get("receiver_email") or "—"
            created_at = format_timestamp(thread.get("created_at"))

            with st.container():
                st.markdown(
                    f"""
                    <div class="thread-card">
                        <b>{subject}</b><br/>
                        <span style="font-size:0.8rem;opacity:0.8;">To: {receiver}</span><br/>
                        <span style="font-size:0.75rem;opacity:0.6;">{created_at}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                col_open, col_delete = st.columns([2, 1])
                with col_open:
                    if st.button(
                        "Open", key=f"open_{thread_id}", use_container_width=True
                    ):
                        st.session_state["selected_thread"] = thread_id
                        st.session_state["workflow_state"] = "viewing_history"
                        st.rerun()
                with col_delete:
                    if st.button(
                        "🗑️", key=f"delete_{thread_id}", use_container_width=True
                    ):
                        st.session_state["confirm_delete_id"] = thread_id
                        st.rerun()

                if st.session_state.get("confirm_delete_id") == thread_id:
                    st.warning(f"Delete thread for '{subject}'? This cannot be undone.")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button(
                            "Confirm Delete",
                            key=f"confirm_delete_{thread_id}",
                            use_container_width=True,
                        ):
                            try:
                                delete_thread(thread_id)
                                st.success("Thread deleted.")
                            except DatabaseError as exc:
                                st.error(f"Could not delete thread: {exc}")
                            st.session_state["confirm_delete_id"] = None
                            if st.session_state.get("selected_thread") == thread_id:
                                st.session_state["selected_thread"] = None
                            st.rerun()
                    with c2:
                        if st.button(
                            "Cancel",
                            key=f"cancel_delete_{thread_id}",
                            use_container_width=True,
                        ):
                            st.session_state["confirm_delete_id"] = None
                            st.rerun()


# --------------------------------------------------------------------------- #
# Email details form (new workflow)
# --------------------------------------------------------------------------- #

def render_email_form() -> None:
    """Render the form for entering new email details and generating a draft."""
    st.markdown("### ✉️ Email Details")

    saved_email = st.session_state.get("saved_sender_email", "")
    saved_password = st.session_state.get("saved_app_password", "")
    if saved_password:
        st.caption(
            f"Using saved app password for **{saved_email}** "
            "(from Gmail Account Setup)."
        )
    else:
        st.caption(
            "Tip: use the **⚙️ Gmail Account Setup** button in the sidebar to "
            "save your app password so you don't have to re-enter it each time."
        )

    with st.form("email_details_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            sender_email = st.text_input(
                "Sender Email", value=saved_email, placeholder="you@example.com"
            )
            sender_password = st.text_input(
                "Password",
                value=saved_password,
                type="password",
                placeholder="App password",
            )
        with col2:
            receiver_email = st.text_input(
                "Receiver Email", placeholder="recipient@example.com"
            )
            subject = st.text_input(
                "Subject", placeholder="Internship Application"
            )

        prompt = st.text_area(
            "Prompt",
            placeholder="Write a professional internship application email.",
            height=120,
        )

        submitted = st.form_submit_button(
            "✨ Generate Draft", use_container_width=True, type="primary"
        )

    if submitted:
        if not all([sender_email, sender_password, receiver_email, subject, prompt]):
            st.warning("Please fill in all fields before generating a draft.")
            return

        with st.spinner("Generating draft..."):
            try:
                result = start_email_workflow(
                    sender_email=sender_email,
                    sender_password=sender_password,
                    receiver_email=receiver_email,
                    subject=subject,
                    prompt=prompt,
                )
            except EmailValidationError as exc:
                st.error(f"Invalid email address: {exc}")
                return
            except LLMGenerationError as exc:
                st.error(f"AI draft generation failed: {exc}")
                return
            except WorkflowError as exc:
                st.error(f"Workflow error: {exc}")
                return
            except DatabaseError as exc:
                st.error(f"Database error: {exc}")
                return

        interrupt_data = result.get("interrupt") or {}
        st.session_state["thread_id"] = result.get("thread_id")
        st.session_state["draft"] = interrupt_data.get("generated_email", "")
        st.session_state["subject"] = interrupt_data.get("subject", subject)
        st.session_state["sender"] = sender_email
        st.session_state["sender_password"] = sender_password
        st.session_state["receiver"] = receiver_email
        st.session_state["prompt"] = prompt
        st.session_state["workflow_state"] = "reviewing"
        st.session_state["approval_status"] = "pending"
        st.session_state["feedback"] = ""
        st.rerun()


# --------------------------------------------------------------------------- #
# Draft review / human-in-the-loop card
# --------------------------------------------------------------------------- #

def render_draft_review() -> None:
    """Render the generated draft, email preview, and HITL review controls."""
    st.markdown("### 🧠 Generated Email Draft")

    with st.container():
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown(f"**Subject:** {st.session_state.get('subject', '')}")
        st.text_area(
            "Generated Email",
            value=st.session_state.get("draft", ""),
            disabled=True,
            height=260,
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Email preview card
    st.markdown("### 👁️ Email Preview")
    st.markdown(
        f"""
        <div class="app-card email-preview">
            <b>From:</b> {st.session_state.get('sender', '')}<br/>
            <b>To:</b> {st.session_state.get('receiver', '')}<br/>
            <b>Subject:</b> {st.session_state.get('subject', '')}<br/><br/>
            {st.session_state.get('draft', '').replace(chr(10), '<br/>')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧑‍⚖️ Human Review")
    with st.container():
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        feedback = st.text_area(
            "Feedback",
            placeholder=(
                "Mention my GitHub profile.\nReduce length.\nMake more formal."
            ),
            height=100,
            key="feedback_input",
        )

        col_edit, col_reject, col_approve = st.columns([1.2, 1, 1.4])
        with col_edit:
            request_changes = st.button(
                "✏️ Request Changes", use_container_width=True
            )
        with col_reject:
            reject = st.button("❌ Reject", use_container_width=True)
        with col_approve:
            approve = st.button(
                "✅ Approve & Send", use_container_width=True, type="primary"
            )
        st.markdown("</div>", unsafe_allow_html=True)

    thread_id = st.session_state.get("thread_id")
    if not thread_id:
        return

    if request_changes:
        if not feedback:
            st.warning("Please provide feedback before requesting changes.")
            return
        with st.spinner("Regenerating draft with your feedback..."):
            try:
                result = resume_workflow(thread_id, "edit", feedback)
            except LLMGenerationError as exc:
                st.error(f"AI regeneration failed: {exc}")
                return
            except WorkflowError as exc:
                st.error(f"Workflow error: {exc}")
                return
            except DatabaseError as exc:
                st.error(f"Database error: {exc}")
                return

        interrupt_data = result.get("interrupt") or {}
        st.session_state["draft"] = interrupt_data.get("generated_email", "")
        st.session_state["subject"] = interrupt_data.get(
            "subject", st.session_state.get("subject", "")
        )
        st.session_state["feedback"] = feedback
        st.session_state["approval_status"] = "edit"
        st.success("New draft generated based on your feedback.")
        st.rerun()

    if approve:
        with st.spinner("Sending email..."):
            try:
                result = resume_workflow(thread_id, "approved")
            except SMTPSendError as exc:
                st.error(f"Failed to send email: {exc}")
                return
            except EmailValidationError as exc:
                st.error(f"Invalid email address: {exc}")
                return
            except WorkflowError as exc:
                st.error(f"Workflow error: {exc}")
                return
            except DatabaseError as exc:
                st.error(f"Database error: {exc}")
                return

        if result.get("sent"):
            st.session_state["workflow_state"] = "sent"
            st.session_state["approval_status"] = "approved"
            st.balloons()
            st.success("✅ Email Sent Successfully")
            st.rerun()
        else:
            st.warning("The workflow did not confirm the email was sent.")

    if reject:
        try:
            resume_workflow(thread_id, "reject")
        except WorkflowError as exc:
            st.error(f"Workflow error: {exc}")
            return
        except DatabaseError as exc:
            st.error(f"Database error: {exc}")
            return
        st.session_state["workflow_state"] = "rejected"
        st.session_state["approval_status"] = "rejected"
        st.warning("Workflow ended. Draft was rejected.")
        st.rerun()


# --------------------------------------------------------------------------- #
# Thread history viewer (when a sidebar thread is opened)
# --------------------------------------------------------------------------- #

def render_thread_viewer(thread_id: str) -> None:
    """Render the full revision history and status for a selected thread."""
    st.markdown("### 🗂️ Thread History")

    try:
        records = load_thread(thread_id)
    except DatabaseError as exc:
        st.error(f"Could not load thread: {exc}")
        return

    if not records:
        st.info("No records found for this thread.")
        return

    latest = records[-1]
    st.markdown(f"**Subject:** {latest.get('subject', '')}")
    st.markdown(
        f"**Status:** {status_badge(latest.get('approved', 'pending'))}",
        unsafe_allow_html=True,
    )
    st.caption(f"Thread ID: `{thread_id}`")

    render_email_timeline(records)

    st.divider()

    for idx, rec in enumerate(records):
        label = f"Revision {idx + 1}" if rec.get("approved") != "approved" else "Final (Sent)"
        with st.expander(f"{label} — {status_badge(rec.get('approved', 'pending'))}", expanded=(idx == len(records) - 1)):
            st.markdown(
                f"""
                <div class="app-card email-preview">
                    <b>From:</b> {rec.get('sender_email', '')}<br/>
                    <b>To:</b> {rec.get('receiver_email', '')}<br/>
                    <b>Subject:</b> {rec.get('subject', '')}<br/>
                    <b>Sent:</b> {format_timestamp(rec.get('sent_time'))}<br/><br/>
                    {(rec.get('generated_email') or '').replace(chr(10), '<br/>')}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if rec.get("feedback"):
                st.caption(f"Feedback given: {rec.get('feedback')}")


# --------------------------------------------------------------------------- #
# Sent-email history page
# --------------------------------------------------------------------------- #

def render_history_page() -> None:
    """Render the list of all sent emails at the bottom of the page."""
    st.markdown("### 📜 Sent Email History")

    try:
        sent_emails = load_all_sent_emails()
    except DatabaseError as exc:
        st.error(f"Could not load sent emails: {exc}")
        return

    if not sent_emails:
        st.caption("No emails have been sent yet.")
        return

    for rec in sent_emails:
        subject = rec.get("subject", "(no subject)")
        receiver = rec.get("receiver_email", "—")
        sent_time = format_timestamp(rec.get("sent_time"))
        thread_id = rec.get("thread_id", "")

        with st.expander(f"📧 {subject} — to {receiver} · {sent_time}"):
            st.markdown(
                f"""
                <div class="app-card email-preview">
                    <b>From:</b> {rec.get('sender_email', '')}<br/>
                    <b>To:</b> {receiver}<br/>
                    <b>Subject:</b> {subject}<br/>
                    <b>Thread ID:</b> <code>{thread_id}</code><br/>
                    <b>Sent:</b> {sent_time}<br/><br/>
                    {(rec.get('generated_email') or '').replace(chr(10), '<br/>')}
                </div>
                """,
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------- #
# Main application
# --------------------------------------------------------------------------- #

def render_header() -> None:
    """Render the gradient hero header."""
    st.markdown(
        """
        <div class="gradient-header">
            <h1>🤖 AI Email Draft Assistant</h1>
            <p>Human-in-the-Loop using LangGraph</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Application entry point."""
    inject_custom_css()
    init_session_state()

    render_header()
    render_metrics()
    st.divider()

    render_sidebar()

    workflow_state = st.session_state.get("workflow_state", "idle")
    selected_thread = st.session_state.get("selected_thread")

    if st.session_state.get("show_settings"):
        render_settings_wizard()
        st.divider()
        if st.button("⬅️ Back"):
            st.session_state["show_settings"] = False
            st.rerun()

    elif selected_thread:
        render_thread_viewer(selected_thread)
        st.divider()
        if st.button("⬅️ Back to New Email"):
            st.session_state["selected_thread"] = None
            st.session_state["workflow_state"] = "idle"
            st.rerun()

    elif workflow_state in ("idle", "rejected"):
        if workflow_state == "rejected":
            st.warning("Previous draft was rejected. Start a new email below.")
        render_email_form()

    elif workflow_state == "reviewing":
        render_draft_review()

    elif workflow_state == "sent":
        st.success("✅ Email Sent Successfully")
        st.markdown(
            f"""
            <div class="app-card email-preview">
                <b>From:</b> {st.session_state.get('sender', '')}<br/>
                <b>To:</b> {st.session_state.get('receiver', '')}<br/>
                <b>Subject:</b> {st.session_state.get('subject', '')}<br/><br/>
                {st.session_state.get('draft', '').replace(chr(10), '<br/>')}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("➕ Compose Another Email", type="primary"):
            reset_workflow_state()
            st.rerun()

    st.divider()
    render_history_page()


if __name__ == "__main__":
    main()