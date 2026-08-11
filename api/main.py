"""
api/main.py

FastAPI service layer. Wires together:
  auth (who is this)
    -> table_access (what are they allowed to ask about)
    -> retrieval + FK expansion + ranking (schema_catalog/*)
    -> access filtering (table_access.filter_ranked_context)
    -> LangGraph self-correction loop (agent/loop.py)
    -> logged response

Every request is logged with user identity, question, generated SQL,
execution result/error, and timing -- this is the foundation the
eval harness and observability work will build on later.

CHANGE: added POST /query/stream. It runs the exact same pipeline as
/query (retrieval -> filtering -> run_agent -> summary) but streams
newline-delimited JSON status events as the LangGraph loop progresses,
so a UI can show live "generating / validating / executing / retrying"
state instead of waiting on one blocking response. /query itself is
untouched.
"""

import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from api.auth import authenticate_user, create_access_token, get_current_user, oauth2_scheme
from api.table_access import filter_ranked_context, has_sufficient_access

from schema_catalog.expand_schema import load_schema_catalog, build_fk_graph
from schema_catalog.rank_schema_context import rank_schema_context
from schema_catalog.retrieve_schema import retrieve_schema
from agent.loop import run_agent
from agent.summarizer import summarize_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("sql_agent")

app = FastAPI(title="Text-to-SQL Agent API")

# Loaded once at startup, reused across requests -- schema and FK graph
# don't change per-request, only per schema update (re-run extraction
# scripts and restart the service when that happens).
_schema_catalog = load_schema_catalog()
_fk_graph = build_fk_graph(_schema_catalog)


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    status: str                  # "success" | "failed" | "access_denied"
    sql: str | None = None
    columns: list[str] | None = None
    rows: list[dict] | None = None
    row_count: int | None = None
    truncated: bool = False
    attempts: int
    error: str | None = None


@app.post("/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    token = create_access_token(data={"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}


@app.get("/me")
def read_current_user(current_user: dict = Depends(get_current_user)):
    return {"username": current_user["username"], "role": current_user["role"]}


@app.post("/query", response_model=QuestionResponse)
def ask_question(request: QuestionRequest, current_user: dict = Depends(get_current_user)):
    start = time.perf_counter()
    question = request.question

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": current_user["id"],
        "username": current_user["username"],
        "role": current_user["role"],
        "question": question,
    }

    # 1. Retrieval + FK expansion + ranking (unchanged core pipeline)
    retrieval_results = retrieve_schema(question, top_k=5)
    ranked_context = rank_schema_context(retrieval_results, _fk_graph)

    # 2. Role-based access filtering -- no-op today under the current
    #    permissive policy, but this is the enforcement point for when
    #    restrictions are added.
    filtered_context = filter_ranked_context(ranked_context, current_user["role"])
    log_entry["access_filtered_tables"] = filtered_context["access_filtered"]

    if not has_sufficient_access(filtered_context):
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_entry.update({"status": "access_denied", "latency_ms": elapsed_ms})
        logger.info(log_entry)
        return QuestionResponse(
            status="access_denied",
            attempts=0,
            error="You don't have access to the data needed to answer this question.",
        )

    # 3. Generate -> validate -> execute, with self-correction retry
    result = run_agent(question, filtered_context)

    elapsed_ms = (time.perf_counter() - start) * 1000
    log_entry.update({
        "status": result["status"],
        "sql": result.get("final_sql"),
        "attempts": result["attempts"],
        "error_history": result["error_history"],
        "latency_ms": round(elapsed_ms, 2),
    })
    logger.info(log_entry)

    if result["status"] == "success":
        exec_result = result["final_result"]
        return QuestionResponse(
            status="success",
            sql=result["final_sql"],
            columns=exec_result["columns"],
            rows=exec_result["rows"],
            row_count=exec_result["row_count"],
            truncated=exec_result["truncated"],
            attempts=result["attempts"],
        )
    else:
        last_error = result["error_history"][-1]["error"] if result["error_history"] else "Unknown error"
        return QuestionResponse(
            status="failed",
            attempts=result["attempts"],
            error=last_error,
        )


@app.post("/query/stream")
def ask_question_stream(request: QuestionRequest, current_user: dict = Depends(get_current_user)):
    """
    Same pipeline as /query, streamed as newline-delimited JSON events.

    Event shapes:
      {"type": "status", "stage": "retrieving_schema"}
      {"type": "status", "stage": "generating"|"validating"|"executing"|"retrying", "attempt": n}
      {"type": "status", "stage": "summarizing"}
      {"type": "result", "status": "success", "summary": str, "columns": [...],
                          "rows": [...], "row_count": n, "truncated": bool, "attempts": n}
      {"type": "result", "status": "failed"|"access_denied", "attempts": n, "error": str}
      {"type": "done"}

    "done" always terminates the stream, whether the run succeeded, failed,
    or raised. The client should stop reading on "done" regardless of what
    came before it.

    Uses plain newline-delimited JSON (application/x-ndjson) rather than
    SSE's "data: ..." framing -- the intended client here is Streamlit's
    own backend making a streamed HTTP request via `requests`, not a
    browser EventSource, so ndjson is simpler on both ends. run_agent()
    itself still runs synchronously inside a background thread; events are
    relayed to the response stream via a queue as the callback fires.
    """
    start = time.perf_counter()
    question = request.question
    event_queue: "queue.Queue[dict]" = queue.Queue()

    def emit(event: dict) -> None:
        event_queue.put(event)

    def worker() -> None:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": current_user["id"],
            "username": current_user["username"],
            "role": current_user["role"],
            "question": question,
        }

        try:
            emit({"type": "status", "stage": "retrieving_schema"})

            retrieval_results = retrieve_schema(question, top_k=5)
            ranked_context = rank_schema_context(retrieval_results, _fk_graph)

            filtered_context = filter_ranked_context(ranked_context, current_user["role"])
            log_entry["access_filtered_tables"] = filtered_context["access_filtered"]

            if not has_sufficient_access(filtered_context):
                elapsed_ms = (time.perf_counter() - start) * 1000
                log_entry.update({"status": "access_denied", "latency_ms": round(elapsed_ms, 2)})
                logger.info(log_entry)
                emit({
                    "type": "result",
                    "status": "access_denied",
                    "attempts": 0,
                    "error": "You don't have access to the data needed to answer this question.",
                })
                return

            result = run_agent(question, filtered_context, on_event=emit)

            log_entry.update({
                "status": result["status"],
                "sql": result.get("final_sql"),
                "attempts": result["attempts"],
                "error_history": result["error_history"],
            })

            if result["status"] == "success":
                exec_result = result["final_result"]

                emit({"type": "status", "stage": "summarizing"})
                try:
                    summary = summarize_result(question, exec_result["columns"], exec_result["rows"])
                except Exception as summary_exc:
                    # Summary is best-effort -- a failure here should never
                    # hide a successful query result from the user.
                    logger.warning(f"summarize_result failed: {summary_exc}")
                    summary = None

                elapsed_ms = (time.perf_counter() - start) * 1000
                log_entry["latency_ms"] = round(elapsed_ms, 2)
                logger.info(log_entry)

                emit({
                    "type": "result",
                    "status": "success",
                    "summary": summary,
                    "sql": result["final_sql"],
                    "columns": exec_result["columns"],
                    "rows": exec_result["rows"],
                    "row_count": exec_result["row_count"],
                    "truncated": exec_result["truncated"],
                    "attempts": result["attempts"],
                })
            else:
                elapsed_ms = (time.perf_counter() - start) * 1000
                log_entry["latency_ms"] = round(elapsed_ms, 2)
                logger.info(log_entry)

                last_error = result["error_history"][-1]["error"] if result["error_history"] else "Unknown error"
                emit({
                    "type": "result",
                    "status": "failed",
                    "attempts": result["attempts"],
                    "error": last_error,
                })
        except Exception as exc:
            logger.exception("Unhandled error in /query/stream worker")
            emit({
                "type": "result",
                "status": "failed",
                "attempts": 0,
                "error": f"Internal error: {exc}",
            })
        finally:
            emit({"type": "done"})

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def event_stream():
        while True:
            event = event_queue.get()
            yield json.dumps(event) + "\n"
            if event.get("type") == "done":
                break

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/health")
def health_check():
    return {"status": "ok"}