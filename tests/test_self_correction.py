"""
tests/test_self_correction.py

End-to-end test for the SQL self-correction mechanism.

This is a TEST-ONLY harness. It does not modify:
    - agent/sql_generator.py
    - agent/sql_validator.py
    - agent/executor.py
    - agent/loop.py
    - the MySQL database

Test flow:

    Attempt 1
        ↓
    generate SQL
        ↓
    validate SQL
        ↓
    deliberately simulate a DB error
        ↓
    build error_context
        ↓
    Attempt 2
        ↓
    generate corrected SQL
        ↓
    validate SQL
        ↓
    execute corrected SQL against MySQL
        ↓
    SUCCESS

The important thing we are proving is that the same error-feedback
mechanism used by agent/loop.py can recover from an execution failure.
"""

from schema_catalog.expand_schema import load_schema_catalog, build_fk_graph
from schema_catalog.rank_schema_context import rank_schema_context
from schema_catalog.retrieve_schema import retrieve_schema

from agent.sql_generator import generate_sql
from agent.sql_validator import validate_sql
from agent.executor import execute_sql


QUESTION = "Which customers rented the most movies?"

MAX_ATTEMPTS = 3


def main():
    """
    Runs the controlled self-correction test.

    Attempt 1 deliberately fails before touching MySQL.

    Attempt 2 uses the real generated error context and, if valid,
    executes the corrected SQL against the real read-only database.
    """

    print("\n--- End-to-end self-correction test ---")

    # ---------------------------------------------------------
    # 1. Build the exact schema context used by the real agent.
    # ---------------------------------------------------------
    schema_catalog = load_schema_catalog()
    fk_graph = build_fk_graph(schema_catalog)

    retrieval_results = retrieve_schema(
        QUESTION,
        top_k=5,
    )

    ranked_context = rank_schema_context(
        retrieval_results,
        fk_graph,
    )

    print(f"\nQuestion: {QUESTION}")
    print(f"Tables: {ranked_context['table_names']}")

    # ---------------------------------------------------------
    # 2. Initial state.
    #
    # This mirrors the important state used by agent/loop.py.
    # ---------------------------------------------------------
    error_history = []
    previous_sql = None

    for attempt in range(1, MAX_ATTEMPTS + 1):

        print(f"\n--- Attempt {attempt} ---")

        # -----------------------------------------------------
        # 3. Build error_context if a previous attempt failed.
        #
        # This is the same information that the real loop sends
        # back to sql_generator.generate_sql().
        # -----------------------------------------------------
        error_context = None

        if error_history:
            last_error = error_history[-1]

            error_context = {
                "previous_sql": last_error["sql"],
                "error_message": last_error["error"],
            }

            print("\nPrevious SQL:")
            print(error_context["previous_sql"])

            print("\nPrevious error:")
            print(error_context["error_message"])

        # -----------------------------------------------------
        # 4. Generate SQL.
        #
        # Attempt 1 has no error context.
        #
        # Attempt 2 receives the simulated DB error.
        # -----------------------------------------------------
        generated = generate_sql(
            QUESTION,
            ranked_context,
            error_context=error_context,
        )

        sql = generated["sql"]
        previous_sql = sql

        print("\nGenerated SQL:")
        print(sql)

        # -----------------------------------------------------
        # 5. Handle CANNOT_ANSWER.
        # -----------------------------------------------------
        if sql == "CANNOT_ANSWER":
            print("\nModel returned CANNOT_ANSWER.")
            print("Self-correction test failed.")
            return

        # -----------------------------------------------------
        # 6. Validate the generated SQL using the real validator.
        # -----------------------------------------------------
        validation = validate_sql(
            sql,
            allowed_tables=ranked_context["table_names"],
        )

        if not validation["valid"]:

            error = (
                f"SQL validation failed: "
                f"{validation['reason']}"
            )

            print(f"\nValidation error: {error}")

            error_history.append({
                "attempt": attempt,
                "sql": sql,
                "error": error,
            })

            if attempt >= MAX_ATTEMPTS:
                break

            continue

        # Use the cleaned SQL from the real validator.
        sql = validation["cleaned_sql"]

        # -----------------------------------------------------
        # 7. DELIBERATE TEST FAILURE.
        #
        # This happens ONLY on attempt 1.
        #
        # We do NOT call MySQL here. We simulate exactly the
        # kind of error that execute_sql() would return.
        # -----------------------------------------------------
        if attempt == 1:

            simulated_error = (
                "OperationalError: "
                "Unknown column 'bad_column' in 'field list'"
            )

            print("\n[TEST] Deliberately simulating database error:")
            print(simulated_error)

            error_history.append({
                "attempt": attempt,
                "sql": sql,
                "error": simulated_error,
            })

            # Continue to attempt 2.
            continue

        # -----------------------------------------------------
        # 8. Attempt 2+:
        #
        # Execute the corrected SQL using the REAL executor.
        # -----------------------------------------------------
        print("\nExecuting corrected SQL against MySQL...")

        result = execute_sql(sql)

        if result["success"]:

            print("\nSUCCESS")
            print(f"Attempts: {attempt}")
            print(f"Rows: {result['row_count']}")
            print(f"Execution time: {result['execution_time_ms']} ms")

            print("\nResult:")
            for row in result["rows"][:3]:
                print(f"  {row}")

            print("\n--- Self-correction test PASSED ---")

            return

        # -----------------------------------------------------
        # 9. If the corrected SQL still fails, preserve the
        # exact database error for another retry.
        # -----------------------------------------------------
        db_error = result["error"]

        error = (
            f"{db_error['type']}: "
            f"{db_error['message']}"
        )

        print(f"\nDatabase error: {error}")

        error_history.append({
            "attempt": attempt,
            "sql": sql,
            "error": error,
        })

    # ---------------------------------------------------------
    # 10. All attempts exhausted.
    # ---------------------------------------------------------
    print("\n--- Self-correction test FAILED ---")
    print(f"Attempts used: {MAX_ATTEMPTS}")

    print("\nError history:")

    for entry in error_history:
        print(f"\nAttempt {entry['attempt']}")
        print(f"SQL: {entry['sql']}")
        print(f"Error: {entry['error']}")


if __name__ == "__main__":
    main()