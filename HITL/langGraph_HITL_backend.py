"""
langGraph_HITL_backend.py

Production-quality backend for an Email Draft Approval system built with
LangGraph Human-in-the-Loop (HITL) workflows.

Workflow
--------
START -> generate_email -> human_review (interrupt) -> [approved] -> send_email -> END
                                        -> [edit]     -> generate_email (loop)
                                        -> [reject]   -> END

Tech stack: LangGraph (latest, interrupt/Command based), LangChain,
ChatMistralAI, MemorySaver, smtplib, sqlite3, dotenv, uuid.

This module is designed to be imported by a Streamlit frontend. It exposes
a compiled `graph`, workflow helper functions, thread-history helpers, and
SQLite persistence helpers.
"""

from __future__ import annotations

import os
import re
import sqlite3
import smtplib
import uuid
import logging
from contextlib import contextmanager
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Iterator, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

# --------------------------------------------------------------------------- #
# Configuration & logging
# --------------------------------------------------------------------------- #

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("langGraph_HITL_backend")

MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    logger.warning(
        "MISTRAL_API_KEY not found in environment. Set it in a .env file "
        "before invoking the graph."
    )

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EmailHistory.db")

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# --------------------------------------------------------------------------- #
# Custom exceptions for readable error handling
# --------------------------------------------------------------------------- #

class EmailValidationError(Exception):
    """Raised when an email address fails validation."""


class SMTPSendError(Exception):
    """Raised when sending an email via SMTP fails."""


class DatabaseError(Exception):
    """Raised when a SQLite operation fails."""


class LLMGenerationError(Exception):
    """Raised when the LLM fails to generate content."""


class WorkflowError(Exception):
    """Raised when the LangGraph workflow encounters an unexpected state."""


# --------------------------------------------------------------------------- #
# SQLite persistence layer
# --------------------------------------------------------------------------- #

@contextmanager
def _get_connection() -> Iterator[sqlite3.Connection]:
    """
    Context manager that yields a SQLite connection with foreign keys
    enabled and commits/rolls back automatically.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise DatabaseError(f"SQLite operation failed: {exc}") from exc
    finally:
        conn.close()


def initialize_database() -> None:
    """
    Create EmailHistory.db (if it does not already exist) along with the
    `threads` and `emails` tables.

    threads: tracks every workflow thread created by the system.
    emails: stores every generated/approved email tied to a thread.
    """
    try:
        with _get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    subject TEXT,
                    receiver_email TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS emails (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    sender_email TEXT NOT NULL,
                    receiver_email TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    generated_email TEXT NOT NULL,
                    approved TEXT NOT NULL,
                    feedback TEXT,
                    sent_time TEXT,
                    FOREIGN KEY (thread_id) REFERENCES threads (thread_id)
                );
                """
            )
        logger.info("Database initialized at %s", DB_PATH)
    except DatabaseError as exc:
        logger.error("Failed to initialize database: %s", exc)
        raise


def create_thread(thread_id: str, subject: str = "", receiver_email: str = "") -> None:
    """Insert a new thread record into the `threads` table."""
    try:
        with _get_connection() as conn:
            conn.execute(
                "INSERT INTO threads (thread_id, created_at, subject, receiver_email) "
                "VALUES (?, ?, ?, ?);",
                (thread_id, datetime.utcnow().isoformat(), subject, receiver_email),
            )
        logger.info("Created thread %s", thread_id)
    except DatabaseError as exc:
        logger.error("Failed to create thread %s: %s", thread_id, exc)
        raise


def save_email(
    thread_id: str,
    sender_email: str,
    receiver_email: str,
    subject: str,
    prompt: str,
    generated_email: str,
    approved: str,
    feedback: str = "",
    sent_time: Optional[str] = None,
) -> None:
    """Persist a single email record (draft or sent) into the `emails` table."""
    try:
        with _get_connection() as conn:
            conn.execute(
                """
                INSERT INTO emails (
                    thread_id, sender_email, receiver_email, subject,
                    prompt, generated_email, approved, feedback, sent_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    thread_id,
                    sender_email,
                    receiver_email,
                    subject,
                    prompt,
                    generated_email,
                    approved,
                    feedback,
                    sent_time,
                ),
            )
        logger.info("Saved email record for thread %s (approved=%s)", thread_id, approved)
    except DatabaseError as exc:
        logger.error("Failed to save email for thread %s: %s", thread_id, exc)
        raise


def fetch_all_threads() -> list[dict[str, Any]]:
    """Return all threads ordered by most recently created first."""
    try:
        with _get_connection() as conn:
            rows = conn.execute(
                "SELECT thread_id, created_at, subject, receiver_email "
                "FROM threads ORDER BY created_at DESC;"
            ).fetchall()
        return [dict(row) for row in rows]
    except DatabaseError as exc:
        logger.error("Failed to fetch threads: %s", exc)
        raise


def fetch_thread(thread_id: str) -> list[dict[str, Any]]:
    """Return all email records associated with a given thread_id."""
    try:
        with _get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM emails WHERE thread_id = ? ORDER BY id ASC;",
                (thread_id,),
            ).fetchall()
        return [dict(row) for row in rows]
    except DatabaseError as exc:
        logger.error("Failed to fetch thread %s: %s", thread_id, exc)
        raise


def fetch_all_emails() -> list[dict[str, Any]]:
    """Return every email record in the database, most recent first."""
    try:
        with _get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM emails ORDER BY id DESC;"
            ).fetchall()
        return [dict(row) for row in rows]
    except DatabaseError as exc:
        logger.error("Failed to fetch all emails: %s", exc)
        raise


def delete_thread(thread_id: str) -> None:
    """Delete a thread and its associated email records."""
    try:
        with _get_connection() as conn:
            conn.execute("DELETE FROM emails WHERE thread_id = ?;", (thread_id,))
            conn.execute("DELETE FROM threads WHERE thread_id = ?;", (thread_id,))
        logger.info("Deleted thread %s", thread_id)
    except DatabaseError as exc:
        logger.error("Failed to delete thread %s: %s", thread_id, exc)
        raise


# --------------------------------------------------------------------------- #
# Thread ID helpers
# --------------------------------------------------------------------------- #

def generate_thread_id() -> str:
    """Generate a new unique thread id."""
    return str(uuid.uuid4())


def get_all_threads() -> list[dict[str, Any]]:
    """Public wrapper returning all threads (see fetch_all_threads)."""
    return fetch_all_threads()


def get_thread_history(thread_id: str) -> list[dict[str, Any]]:
    """Public wrapper returning all email history for a thread."""
    return fetch_thread(thread_id)


# --------------------------------------------------------------------------- #
# Graph state definition
# --------------------------------------------------------------------------- #

class EmailState(TypedDict):
    """Shared state passed between LangGraph nodes."""
    sender_email: str
    sender_password: str
    receiver_email: str
    subject: str
    prompt: str
    generated_email: str
    feedback: str
    approval: str
    sent: bool
    thread_id: str


# --------------------------------------------------------------------------- #
# LLM instance
# --------------------------------------------------------------------------- #

def _build_llm() -> ChatMistralAI:
    """Instantiate the ChatMistralAI model used to draft emails."""
    if not MISTRAL_API_KEY:
        raise LLMGenerationError(
            "MISTRAL_API_KEY is not set. Please add it to your .env file."
        )
    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.7,
        api_key=MISTRAL_API_KEY,
    )


llm = _build_llm() if MISTRAL_API_KEY else None


# --------------------------------------------------------------------------- #
# Node 1: generate_email
# --------------------------------------------------------------------------- #

def generate_email(state: EmailState) -> dict[str, Any]:
    """
    Generate (or regenerate) a professional email draft using ChatMistralAI.

    If `feedback` is present in the state (from a previous human review
    round), the LLM is instructed to revise the draft accordingly.
    """
    global llm
    if llm is None:
        llm = _build_llm()

    receiver = state.get("receiver_email", "")
    subject = state.get("subject", "")
    user_prompt = state.get("prompt", "")
    feedback = state.get("feedback", "")

    system_instructions = (
        "You are a professional email writing assistant. "
        "Write clear, professional, well-structured email body text only. "
        "Do not use markdown formatting, do not use emojis, "
        "do not wrap the output in triple backticks, "
        "and do not include a subject line — return only the email body."
    )

    human_instructions = (
        f"Recipient: {receiver}\n"
        f"Subject: {subject}\n"
        f"User request: {user_prompt}\n"
    )
    if feedback:
        human_instructions += (
            f"\nPrevious draft feedback to incorporate:\n{feedback}\n"
            "Please revise the email to address this feedback."
        )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_instructions),
                HumanMessage(content=human_instructions),
            ]
        )
        generated_text = str(response.content).strip()
        if not generated_text:
            raise LLMGenerationError("LLM returned an empty email draft.")
    except Exception as exc:  # noqa: BLE001 - surfaced as readable error
        logger.error("LLM generation failed: %s", exc)
        raise LLMGenerationError(f"Failed to generate email draft: {exc}") from exc

    # Persist the draft (not yet approved) for audit/history purposes.
    try:
        save_email(
            thread_id=state["thread_id"],
            sender_email=state.get("sender_email", ""),
            receiver_email=receiver,
            subject=subject,
            prompt=user_prompt,
            generated_email=generated_text,
            approved="pending",
            feedback=feedback,
            sent_time=None,
        )
    except DatabaseError as exc:
        # Draft generation should not hard-fail the workflow on a logging
        # error, but we surface it clearly.
        logger.warning("Could not save draft to database: %s", exc)

    return {"generated_email": generated_text, "feedback": ""}


# --------------------------------------------------------------------------- #
# Node 2: human_review
# --------------------------------------------------------------------------- #

def human_review(state: EmailState) -> dict[str, Any]:
    """
    Pause the graph and wait for human input via LangGraph's `interrupt`.

    The interrupt payload exposes the generated email, subject, and
    thread id so the frontend can render a review UI. Execution resumes
    when the frontend calls `graph.invoke(Command(resume=...), config)`.

    Expected resume payload (a dict):
        {"approval": "approved" | "edit" | "reject", "feedback": "<str>"}
    """
    review_payload = {
        "generated_email": state.get("generated_email", ""),
        "subject": state.get("subject", ""),
        "thread_id": state.get("thread_id", ""),
    }

    human_response: dict[str, Any] = interrupt(review_payload)

    approval = human_response.get("approval", "reject")
    feedback = human_response.get("feedback", "")

    return {"approval": approval, "feedback": feedback}


# --------------------------------------------------------------------------- #
# Node 3: send_email
# --------------------------------------------------------------------------- #

def _validate_email_address(address: str) -> None:
    """Raise EmailValidationError if `address` is not a valid-looking email."""
    if not address or not EMAIL_REGEX.match(address):
        raise EmailValidationError(f"Invalid email address: '{address}'")


def send_email(state: EmailState) -> dict[str, Any]:
    """
    Send the approved email via SMTP and persist the final record in
    SQLite. Only reached when approval == "approved".
    """
    sender_email = state.get("sender_email", "")
    sender_password = state.get("sender_password", "")
    receiver_email = state.get("receiver_email", "")
    subject = state.get("subject", "")
    body = state.get("generated_email", "")
    thread_id = state.get("thread_id", "")

    try:
        _validate_email_address(sender_email)
        _validate_email_address(receiver_email)
    except EmailValidationError as exc:
        logger.error("Email validation failed: %s", exc)
        raise

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
    except smtplib.SMTPAuthenticationError as exc:
        logger.error("SMTP authentication failed: %s", exc)
        raise SMTPSendError(
            "SMTP login failed. Check sender email and app password."
        ) from exc
    except smtplib.SMTPException as exc:
        logger.error("SMTP send failed: %s", exc)
        raise SMTPSendError(f"Failed to send email: {exc}") from exc
    except OSError as exc:
        logger.error("Network error while sending email: %s", exc)
        raise SMTPSendError(f"Network error while sending email: {exc}") from exc

    sent_time = datetime.utcnow().isoformat()

    try:
        save_email(
            thread_id=thread_id,
            sender_email=sender_email,
            receiver_email=receiver_email,
            subject=subject,
            prompt=state.get("prompt", ""),
            generated_email=body,
            approved="approved",
            feedback="",
            sent_time=sent_time,
        )
    except DatabaseError as exc:
        # The email was sent even if saving history fails; surface a
        # readable warning but do not lose the fact that it was sent.
        logger.warning("Email sent but failed to save history: %s", exc)

    logger.info("Email sent successfully for thread %s", thread_id)
    return {"sent": True}


# --------------------------------------------------------------------------- #
# Conditional routing
# --------------------------------------------------------------------------- #

def route_after_review(state: EmailState) -> str:
    """
    Decide the next node after human_review based on `approval`.

    approved -> send_email
    edit     -> generate_email (loop back for a revised draft)
    anything else (e.g. reject) -> END
    """
    approval = state.get("approval", "")
    if approval == "approved":
        return "send_email"
    if approval == "edit":
        return "generate_email"
    return END


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #

def build_graph() -> Any:
    """
    Construct and compile the LangGraph StateGraph with MemorySaver.
    
    LANGGRAPH CONCEPTS:
    - StateGraph: Defines the nodes and edges of the workflow. The state schema is `EmailState`.
    - Nodes: Python functions that receive the current state and return state updates.
    - Edges: Define the flow of execution from one node to the next.
    - Conditional Edges: Route execution dynamically based on the state (e.g., approval status).
    - Checkpointer (MemorySaver): Saves the state of the graph after each step. Crucial for HITL because the graph execution pauses and needs to be restored later using the thread_id.
    """
    workflow = StateGraph(EmailState)

    # Add all nodes to the graph
    workflow.add_node("generate_email", generate_email)
    workflow.add_node("human_review", human_review)
    workflow.add_node("send_email", send_email)

    # Define the workflow structure
    workflow.add_edge(START, "generate_email")
    workflow.add_edge("generate_email", "human_review")
    
    # After human_review, dynamically decide where to go next based on `route_after_review`
    workflow.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "send_email": "send_email",
            "generate_email": "generate_email",
            END: END,
        },
    )
    workflow.add_edge("send_email", END)

    # Compile the graph with a checkpointer to enable state persistence and `interrupt` functionality
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


graph = build_graph()


# --------------------------------------------------------------------------- #
# High-level workflow helper functions (for the Streamlit frontend)
# --------------------------------------------------------------------------- #

def start_email_workflow(
    sender_email: str,
    sender_password: str,
    receiver_email: str,
    subject: str,
    prompt: str,
) -> dict[str, Any]:
    """
    Start a brand-new email workflow on a fresh thread.

    Returns a dict containing the thread_id and the interrupt payload
    (generated email awaiting human review) so the frontend can display
    it for approval/edit/reject.
    """
    try:
        _validate_email_address(sender_email)
        _validate_email_address(receiver_email)
    except EmailValidationError as exc:
        raise WorkflowError(str(exc)) from exc

    thread_id = generate_thread_id()

    try:
        create_thread(thread_id, subject=subject, receiver_email=receiver_email)
    except DatabaseError as exc:
        raise WorkflowError(f"Could not create thread: {exc}") from exc

    initial_state: EmailState = {
        "sender_email": sender_email,
        "sender_password": sender_password,
        "receiver_email": receiver_email,
        "subject": subject,
        "prompt": prompt,
        "generated_email": "",
        "feedback": "",
        "approval": "",
        "sent": False,
        "thread_id": thread_id,
    }

    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph.invoke(initial_state, config=config)
    except LLMGenerationError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Workflow error while starting thread %s: %s", thread_id, exc)
        raise WorkflowError(f"Failed to start workflow: {exc}") from exc

    interrupt_payload = _extract_interrupt_payload(result)
    return {"thread_id": thread_id, "interrupt": interrupt_payload}


def resume_workflow(
    thread_id: str,
    approval: str,
    feedback: str = "",
) -> dict[str, Any]:
    """
    Resume a paused workflow after human review.

    approval:
        "approved" -> proceeds to send_email
        "edit"     -> loops back to generate_email with the given feedback
        "reject"   -> ends the workflow

    Returns a dict with either:
        {"thread_id": ..., "interrupt": {...}}   if paused again (edit loop)
        {"thread_id": ..., "sent": True/False}   if workflow finished
    """
    config = {"configurable": {"thread_id": thread_id}}

    resume_payload = {"approval": approval, "feedback": feedback}

    try:
        result = graph.invoke(Command(resume=resume_payload), config=config)
    except SMTPSendError:
        raise
    except EmailValidationError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Workflow error while resuming thread %s: %s", thread_id, exc)
        raise WorkflowError(f"Failed to resume workflow: {exc}") from exc

    interrupt_payload = _extract_interrupt_payload(result)
    if interrupt_payload is not None:
        return {"thread_id": thread_id, "interrupt": interrupt_payload}

    return {"thread_id": thread_id, "sent": bool(result.get("sent", False))}


def _extract_interrupt_payload(result: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Given the raw output of graph.invoke(...), extract the interrupt
    payload if the graph is currently paused, otherwise return None.
    """
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    # LangGraph returns a tuple/list of Interrupt objects; take the first.
    first = interrupts[0]
    return first.value if hasattr(first, "value") else first


# --------------------------------------------------------------------------- #
# Thread history / search helpers (for the Streamlit frontend)
# --------------------------------------------------------------------------- #

def list_threads() -> list[dict[str, Any]]:
    """Return a list of all threads for display in the sidebar."""
    return get_all_threads()


def load_thread(thread_id: str) -> list[dict[str, Any]]:
    """Load the full email history for a specific thread."""
    return get_thread_history(thread_id)


def load_all_sent_emails() -> list[dict[str, Any]]:
    """Return every email record marked as 'approved' (i.e. sent)."""
    all_emails = fetch_all_emails()
    return [row for row in all_emails if row.get("approved") == "approved"]


def search_threads(keyword: str) -> list[dict[str, Any]]:
    """
    Search across subject and generated_email text for a keyword,
    returning matching email records (across all threads).
    """
    if not keyword:
        return []
    try:
        with _get_connection() as conn:
            like_pattern = f"%{keyword}%"
            rows = conn.execute(
                """
                SELECT * FROM emails
                WHERE subject LIKE ? OR generated_email LIKE ?
                ORDER BY id DESC;
                """,
                (like_pattern, like_pattern),
            ).fetchall()
        return [dict(row) for row in rows]
    except DatabaseError as exc:
        logger.error("Failed to search threads for '%s': %s", keyword, exc)
        raise


# --------------------------------------------------------------------------- #
# Module bootstrap
# --------------------------------------------------------------------------- #

initialize_database()


if __name__ == "__main__":
    # Simple manual smoke test (requires a valid .env with MISTRAL_API_KEY
    # and real SMTP credentials to actually send mail).
    logger.info("langGraph_HITL_backend module loaded. Graph object: %s", graph)