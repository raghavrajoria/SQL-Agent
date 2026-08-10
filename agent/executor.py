"""
agent/executor.py

Executes validated SQL against MySQL using the read-only credential
(DB_URL_READONLY). This is the ONLY module that should ever open a DB
connection for running generated queries -- sql_generator.py and
sql_validator.py never touch the database directly, by design.

Two independent safety layers are enforced by the time SQL reaches here:
1. Application-level validation (sql_validator.py) -- already passed.
2. DB-level permissions (sql_agent_readonly, GRANT SELECT only) -- enforced
   by MySQL itself on every query this module runs, verified manually.

This module adds a THIRD protection specific to execution: a query
timeout, so a slow/expensive generated query (e.g. an accidental cross
join on a large production table) can't hang the whole pipeline.

Errors are captured and returned as structured data, not raised, because
the self-correction loop needs the exact error message to feed back to
the LLM for retry -- a raised exception would need to be caught and
reformatted anyway, so returning structured results directly avoids that
duplication.
"""

import os
import time
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

load_dotenv()

QUERY_TIMEOUT_SECONDS = 5   # hard cap per query -- generated SQL is
                              # untrusted in terms of cost, not just safety
MAX_ROWS_RETURNED = 500      # caps result size sent back through the
                              # pipeline / eventually to the UI -- a
                              # "SELECT * FROM film" style query on a real
                              # 200-table production DB could otherwise
                              # return an unbounded result set

_engine = None  # lazily created, reused across calls -- avoids reconnecting per query


def get_engine():
    global _engine
    if _engine is None:
        db_url = os.environ["DB_URL_READONLY"]
        _engine = create_engine(
            db_url,
            pool_pre_ping=True,   # detects stale connections before use
            connect_args={"connect_timeout": QUERY_TIMEOUT_SECONDS},
        )
    return _engine


def execute_sql(sql: str) -> dict:
    """
    Args:
        sql: validated SQL (must already have passed sql_validator.validate_sql)

    Returns:
        {
          "success": bool,
          "columns": list[str] or None,
          "rows": list[dict] or None,      # capped at MAX_ROWS_RETURNED
          "row_count": int or None,          # actual rows returned (post-cap)
          "truncated": bool,                  # True if MAX_ROWS_RETURNED was hit
          "execution_time_ms": float,
          "error": {
              "type": str,       # exception class name, e.g. "OperationalError"
              "message": str      # the raw DB error message -- this is what
                                     # gets fed back to the LLM in the
                                     # self-correction loop, so keep it intact,
                                     # don't sanitize/shorten it here
          } or None
        }
    """
    engine = get_engine()
    start = time.perf_counter()

    try:
        with engine.connect() as conn:
            # Per-session statement timeout, enforced by MySQL itself --
            # protects against expensive queries independent of the
            # connect_timeout above, which only covers connection setup.
            conn.execute(text(f"SET SESSION MAX_EXECUTION_TIME={QUERY_TIMEOUT_SECONDS * 1000}"))

            result = conn.execute(text(sql))
            columns = list(result.keys())

            rows = []
            truncated = False
            for i, row in enumerate(result):
                if i >= MAX_ROWS_RETURNED:
                    truncated = True
                    break
                rows.append(dict(zip(columns, row)))

            elapsed_ms = (time.perf_counter() - start) * 1000

            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "truncated": truncated,
                "execution_time_ms": round(elapsed_ms, 2),
                "error": None,
            }

    except SQLAlchemyError as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        # Unwrap SQLAlchemy's wrapper to get the underlying DB error message --
        # that's the actionable part for the self-correction loop, the
        # SQLAlchemy wrapper text adds noise on top of it.
        raw_message = str(e.orig) if hasattr(e, "orig") and e.orig else str(e)

        return {
            "success": False,
            "columns": None,
            "rows": None,
            "row_count": None,
            "truncated": False,
            "execution_time_ms": round(elapsed_ms, 2),
            "error": {
                "type": type(e).__name__,
                "message": raw_message,
            },
        }


if __name__ == "__main__":
    test_queries = [
        "SELECT customer_id, first_name, last_name FROM customer LIMIT 3",
        "SELECT * FROM nonexistent_table",  # should produce a clean error
        "SELECT c.customer_id, COUNT(r.rental_id) AS rental_count "
        "FROM customer c JOIN rental r ON c.customer_id = r.customer_id "
        "GROUP BY c.customer_id ORDER BY rental_count DESC LIMIT 1",
    ]

    for sql in test_queries:
        print(f"SQL: {sql}")
        result = execute_sql(sql)
        if result["success"]:
            print(f"  rows: {result['row_count']}, time: {result['execution_time_ms']}ms")
            for row in result["rows"]:
                print(f"    {row}")
        else:
            print(f"  ERROR [{result['error']['type']}]: {result['error']['message']}")
        print()