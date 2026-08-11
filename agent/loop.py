"""
agent/loop.py

The self-correction agent, built as a LangGraph state machine.

This is the piece that makes the pipeline genuinely agentic, not just an
LLM feature: it generates SQL, executes it, and on failure feeds the exact
error back to the model for another attempt -- up to a retry cap. This is
a cycle with conditional branching, which is exactly what LangGraph is
built to model (a plain linear LangChain chain can't represent a loop like
this cleanly).

Graph shape:

    generate --> validate --+--> (invalid) --> retry_check --+--> generate (loop)
                             |                                 |
                             +--> (valid) --> execute          +--> END (failed, retries exhausted)
                                                  |
                                       +----------+----------+
                                       |                     |
                                  (success)              (db error)
                                       |                     |
                                      END              retry_check (same as above)

Both validation failures and execution failures are treated as "attempt
failed, retry with this error" -- the retry logic doesn't care which stage
failed, it just needs an error message to feed back to generation.

CHANGE (streaming support): run_agent() now accepts an optional on_event
callback. Each node fires it with a small dict describing what it's about
to do -- {"type": "status", "stage": ..., "attempt": ...} -- before doing
the work. This is the only change; graph shape, retry logic, and state
fields are untouched, so run_agent(question, ranked_context) with no
callback behaves exactly as before.
"""

from typing import TypedDict, Optional, Callable
from langgraph.graph import StateGraph, END

from agent.sql_generator import generate_sql
from agent.sql_validator import validate_sql
from agent.executor import execute_sql

MAX_ATTEMPTS = 3  # total generation attempts before giving up, including the first


class AgentState(TypedDict):
    question: str
    ranked_context: dict
    attempt: int
    sql: Optional[str]
    error_history: list[dict]     # [{"attempt": n, "sql": ..., "error": ...}, ...]
    status: str                    # "in_progress" | "success" | "failed"
    final_result: Optional[dict]   # execute_sql() output, once successful
    final_sql: Optional[str]
    _on_event: Optional[Callable[[dict], None]]  # not part of the "real" agent
                                                   # state, just a pass-through hook.
                                                   # Never read by generate/validate/
                                                   # execute logic itself.


def _emit(state: AgentState, event: dict) -> None:
    callback = state.get("_on_event")
    if callback is not None:
        callback(event)


def generate_node(state: AgentState) -> AgentState:
    _emit(state, {"type": "status", "stage": "generating", "attempt": state["attempt"] + 1})

    error_context = None
    if state["error_history"]:
        last_error = state["error_history"][-1]
        error_context = {
            "previous_sql": last_error["sql"],
            "error_message": last_error["error"],
        }

    result = generate_sql(state["question"], state["ranked_context"], error_context=error_context)

    state["sql"] = result["sql"]
    state["attempt"] += 1
    return state


def validate_node(state: AgentState) -> AgentState:
    _emit(state, {"type": "status", "stage": "validating", "attempt": state["attempt"]})

    sql = state["sql"]

    if sql == "CANNOT_ANSWER":
        state["status"] = "failed"
        state["error_history"].append({
            "attempt": state["attempt"],
            "sql": sql,
            "error": "Model indicated the question cannot be answered with the available schema.",
        })
        return state

    validation = validate_sql(sql, allowed_tables=state["ranked_context"]["table_names"])

    if not validation["valid"]:
        state["error_history"].append({
            "attempt": state["attempt"],
            "sql": sql,
            "error": f"SQL validation failed: {validation['reason']}",
        })
        state["status"] = "in_progress"  # will route to retry_check
    else:
        state["sql"] = validation["cleaned_sql"]
        state["status"] = "validated"

    return state


def execute_node(state: AgentState) -> AgentState:
    _emit(state, {"type": "status", "stage": "executing", "attempt": state["attempt"]})

    result = execute_sql(state["sql"])

    if result["success"]:
        state["status"] = "success"
        state["final_result"] = result
        state["final_sql"] = state["sql"]
    else:
        state["error_history"].append({
            "attempt": state["attempt"],
            "sql": state["sql"],
            "error": f"{result['error']['type']}: {result['error']['message']}",
        })
        state["status"] = "in_progress"  # will route to retry_check

    return state


def route_after_validate(state: AgentState) -> str:
    if state["status"] == "validated":
        return "execute"
    if state["status"] == "failed":
        return "end"  # CANNOT_ANSWER -- no point retrying, model already said it can't
    return "retry_check"


def route_after_execute(state: AgentState) -> str:
    if state["status"] == "success":
        return "end"
    return "retry_check"


def route_retry(state: AgentState) -> str:
    if state["attempt"] >= MAX_ATTEMPTS:
        state["status"] = "failed"
        return "end"
    _emit(state, {"type": "status", "stage": "retrying", "attempt": state["attempt"] + 1})
    return "generate"

def retry_check_node(state: AgentState) -> AgentState:
    """
    Pass-through node used only to give LangGraph a concrete node from which
    the retry decision can be routed.

    The actual retry decision is handled by route_retry(). This node does
    not modify the state.
    """
    return state

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("execute", execute_node)
    graph.add_node("retry_check", retry_check_node)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "validate")

    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"execute": "execute", "retry_check": "retry_check", "end": END},
    )

    # route_retry decides retry vs. give up; wire both outcomes explicitly
    # rather than letting the graph library guess.
    graph.add_conditional_edges(
        "execute",
        route_after_execute,
        {"end": END, "retry_check": "retry_check"},
    )

    graph.add_conditional_edges(
        "retry_check",
        route_retry,
        {"generate": "generate", "end": END},
    )

    return graph.compile()
_compiled_graph = None


def run_agent(question: str, ranked_context: dict, on_event: Optional[Callable[[dict], None]] = None) -> dict:
    """
    Runs the full self-correction loop for a question against a ranked
    schema context (already produced by retrieval + FK expansion + ranking
    upstream -- this function does not re-run retrieval).

    Args:
        on_event: optional callback fired synchronously from inside node
            functions as the graph progresses, e.g.
            {"type": "status", "stage": "generating", "attempt": 1}.
            Intended for streaming endpoints to relay live progress to a
            client. If omitted, behavior is identical to before this change.

    Returns:
        {
          "status": "success" | "failed",
          "final_sql": str or None,
          "final_result": dict or None,   # execute_sql() output if successful
          "attempts": int,
          "error_history": list[dict]       # full trail of every failed attempt,
                                               # useful for debugging/logging
        }
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()

    initial_state: AgentState = {
        "question": question,
        "ranked_context": ranked_context,
        "attempt": 0,
        "sql": None,
        "error_history": [],
        "status": "in_progress",
        "final_result": None,
        "final_sql": None,
        "_on_event": on_event,
    }

    final_state = _compiled_graph.invoke(initial_state)

    return {
        "status": final_state["status"],
        "final_sql": final_state.get("final_sql"),
        "final_result": final_state.get("final_result"),
        "attempts": final_state["attempt"],
        "error_history": final_state["error_history"],
    }


if __name__ == "__main__":
    from schema_catalog.expand_schema import load_schema_catalog, build_fk_graph
    from schema_catalog.rank_schema_context import rank_schema_context
    from schema_catalog.retrieve_schema import retrieve_schema

    schema_catalog = load_schema_catalog()
    fk_graph = build_fk_graph(schema_catalog)

    test_questions = [
        "Which customer paid the least and what was the total amount paid for renting movies?"
    ]

    for question in test_questions:
        retrieval_results = retrieve_schema(question, top_k=3)
        ranked_context = rank_schema_context(retrieval_results, fk_graph)

        result = run_agent(question, ranked_context)

        print(f"\nQ: {question}")
        print(f"  status: {result['status']}, attempts: {result['attempts']}")
        if result["status"] == "success":
            print(f"  SQL: {result['final_sql']}")
            print(f"  rows: {result['final_result']['row_count']}")
            for row in result["final_result"]["rows"][:3]:
                print(f"    {row}")
        else:
            print(f"  error trail:")
            for e in result["error_history"]:
                print(f"    attempt {e['attempt']}: {e['error']}")

    # --- Deliberate failure test: force a retry ---
    # Give the agent a ranked context missing tables it actually needs.
    # For the revenue question, only "film" (no rental/payment/inventory) --
    # the model can't answer this correctly without those tables, so it
    # should either produce SQL referencing an out-of-scope table
    # (caught by validation) or invalid SQL (caught by execution), forcing
    # at least one retry. This proves the retry mechanism actually fires,
    # not just that the happy path works.
    print("\n--- Deliberate failure test (forces retry) ---")
    broken_context = {
        "table_names": ["film"],
        "join_paths": [],
        "ranked_tables": [{"table_name": "film", "score": 1.0, "hops": 0}],
        "truncated": False,
    }
    broken_question = "Which films generated the most rental revenue?"
    result = run_agent(broken_question, broken_context)
    print(f"Q: {broken_question} (with incomplete schema context: only 'film')")
    print(f"  status: {result['status']}, attempts: {result['attempts']}")
    print(f"  error trail:")
    for e in result["error_history"]:
        print(f"    attempt {e['attempt']}: {e['error'][:150]}")
    if result["status"] == "success":
        print(f"  (unexpectedly succeeded -- model may have answered a narrower question)")
        print(f"  SQL: {result['final_sql']}")