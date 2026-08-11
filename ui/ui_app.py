"""
ui/app.py

Streamlit client for the Text-to-SQL FastAPI service.

The UI does not connect to MySQL, call Groq, retrieve schema, or run SQL.
All authentication and query execution are handled by FastAPI. This client
only calls /auth/login, /me, and /query/stream.

CHANGE from the previous version: queries now go to /query/stream instead
of /query. Progress is shown live via st.status() as a Claude-style
"thinking" trace (retrieving schema -> generating SQL -> validating ->
executing -> retrying if needed -> summarizing), then collapses into a
final chat bubble with a plain-English summary and the result table.
Generated SQL is intentionally not displayed.
"""

import json
import os

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 60
STREAM_READ_TIMEOUT = 120  # self-correction can take a few LLM round trips

STAGE_LABELS = {
    "retrieving_schema": "Reading the schema...",
    "generating": "Writing SQL (attempt {attempt})...",
    "validating": "Checking the query...",
    "executing": "Running it against the database...",
    "retrying": "That didn't work, trying again (attempt {attempt})...",
    "summarizing": "Putting the answer together...",
}


def api_url(path: str) -> str:
    return f"{st.session_state.api_base_url.rstrip('/')}{path}"


def login(username: str, password: str) -> tuple[bool, str]:
    try:
        response = requests.post(
            api_url("/auth/login"),
            data={"username": username, "password": password},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach the FastAPI service: {exc}"

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", "Login failed.")
        except ValueError:
            detail = "Login failed."
        return False, detail

    try:
        payload = response.json()
    except ValueError:
        return False, "FastAPI returned an invalid login response."

    token = payload.get("access_token")
    if not token:
        return False, "Login response did not contain an access token."

    st.session_state.token = token
    return True, ""


def get_current_user() -> tuple[bool, dict | str]:
    try:
        response = requests.get(
            api_url("/me"),
            headers={"Authorization": f"Bearer {st.session_state.token}"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach the FastAPI service: {exc}"

    if response.status_code == 401:
        return False, "Session expired. Please log in again."

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", "Could not verify the user.")
        except ValueError:
            detail = "Could not verify the user."
        return False, detail

    try:
        return True, response.json()
    except ValueError:
        return False, "FastAPI returned an invalid /me response."


def stream_question(question: str, status_box) -> dict:
    """
    Posts to /query/stream and consumes newline-delimited JSON events as
    they arrive, updating the given st.status() box live. Returns the
    final "result" event dict, or a synthetic failed/error dict if the
    stream broke before one arrived.
    """
    try:
        response = requests.post(
            api_url("/query/stream"),
            json={"question": question},
            headers={"Authorization": f"Bearer {st.session_state.token}"},
            timeout=(REQUEST_TIMEOUT, STREAM_READ_TIMEOUT),
            stream=True,
        )
    except requests.RequestException as exc:
        return {"status": "failed", "attempts": 0, "error": f"Could not reach the FastAPI service: {exc}"}

    if response.status_code == 401:
        return {"status": "failed", "attempts": 0, "error": "__SESSION_EXPIRED__"}

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", "Query request failed.")
        except ValueError:
            detail = f"Query request failed (HTTP {response.status_code})."
        return {"status": "failed", "attempts": 0, "error": detail}

    result_event = None

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue

        event_type = event.get("type")

        if event_type == "status":
            stage = event.get("stage", "")
            label = STAGE_LABELS.get(stage, stage)
            if "{attempt}" in label:
                label = label.format(attempt=event.get("attempt", "?"))
            status_box.update(label=label, state="running")

        elif event_type == "result":
            result_event = event

        elif event_type == "done":
            break

    if result_event is None:
        result_event = {"status": "failed", "attempts": 0, "error": "Stream ended without a result."}

    return result_event


def initialize_session() -> None:
    defaults = {
        "api_base_url": DEFAULT_API_URL,
        "token": None,
        "user": None,
        "messages": [],  # [{"role": "user"|"assistant", "content": str, "table": {...} | None}, ...]
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_login() -> None:
    st.title("Text-to-SQL Agent")
    st.caption("Authenticated natural-language SQL interface")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")

    if submitted:
        if not username or not password:
            st.error("Username and password are required.")
            return

        success, error = login(username, password)
        if not success:
            st.error(error)
            return

        verified, user_or_error = get_current_user()
        if not verified:
            st.session_state.token = None
            st.error(str(user_or_error))
            return

        st.session_state.user = user_or_error
        st.rerun()


def render_assistant_message(content: str, table: dict | None) -> None:
    st.markdown(content)
    if table and table.get("rows"):
        dataframe = pd.DataFrame(table["rows"], columns=table.get("columns") or None)
        st.dataframe(dataframe, use_container_width=True)
        if table.get("truncated"):
            st.caption("Results were truncated -- there's more data than shown here.")


def show_chat_interface() -> None:
    user = st.session_state.user

    with st.sidebar:
        st.subheader("Session")
        st.write(f"**User:** {user.get('username', 'Unknown')}")
        st.write(f"**Role:** {user.get('role', 'Unknown')}")

        if st.button("Log out", use_container_width=True):
            st.session_state.token = None
            st.session_state.user = None
            st.session_state.messages = []
            st.rerun()

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.title("Text-to-SQL Agent")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message["content"], message.get("table"))
            else:
                st.markdown(message["content"])

    question = st.chat_input("Ask a question about the data...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question, "table": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("Thinking...", expanded=True) as status_box:
            result = stream_question(question, status_box)

        if result.get("error") == "__SESSION_EXPIRED__":
            status_box.update(label="Session expired", state="error")
            st.session_state.token = None
            st.session_state.user = None
            st.error("Session expired. Please log in again.")
            st.session_state.messages.pop()  # drop the user turn we can't answer
            st.rerun()

        status = result.get("status")

        if status == "success":
            status_box.update(label="Done", state="complete", expanded=False)

            summary = result.get("summary") or "Here's what I found:"
            table = {
                "columns": result.get("columns") or [],
                "rows": result.get("rows") or [],
                "truncated": result.get("truncated", False),
            }

            render_assistant_message(summary, table)
            st.session_state.messages.append({"role": "assistant", "content": summary, "table": table})

        elif status == "access_denied":
            status_box.update(label="Access denied", state="error", expanded=False)
            error_text = result.get("error") or "You don't have access to the data needed for that question."
            st.markdown(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text, "table": None})

        else:  # failed, or anything unexpected
            status_box.update(label="Couldn't complete that", state="error", expanded=False)
            error_text = result.get("error") or "Something went wrong answering that question."
            attempts = result.get("attempts", 0)
            content = f"I couldn't get a reliable answer after {attempts} attempt(s).\n\n{error_text}"
            st.markdown(content)
            st.session_state.messages.append({"role": "assistant", "content": content, "table": None})


def main() -> None:
    st.set_page_config(
        page_title="Text-to-SQL Agent",
        page_icon="SQL",
        layout="wide",
    )

    initialize_session()

    with st.sidebar:
        st.text_input(
            "FastAPI URL",
            key="api_base_url",
            help="Base URL of the FastAPI service.",
        )

    if not st.session_state.token:
        show_login()
        return

    verified, user_or_error = get_current_user()
    if not verified:
        st.session_state.token = None
        st.session_state.user = None
        st.session_state.messages = []
        show_login()
        return

    st.session_state.user = user_or_error
    show_chat_interface()


if __name__ == "__main__":
    main()