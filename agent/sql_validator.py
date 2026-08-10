"""
agent/sql_validator.py

Application-level guardrail: rejects any generated SQL that isn't a single,
clean SELECT statement, BEFORE it ever reaches the database.

This is layer 1 of two independent guardrails (per your architecture
decision). Layer 2 is the DB-level read-only role (GRANT SELECT only) --
still to be set up on the MySQL side. This module does NOT replace that;
it exists so bad queries get rejected early with a clear error message,
rather than relying solely on the DB permission to silently fail them.
Do not skip setting up the DB-level read-only role just because this
module exists -- if this module has a bug, the DB permission is the only
thing standing between a malformed query and real data.

This module does not execute anything. It only classifies: valid or not,
and why. Execution is the next, separate piece.
"""

import re

# Keywords that must never appear in generated SQL, regardless of position.
# Checked as whole words (not substrings) to avoid false positives like
# rejecting a column named "created_at" because it contains "create".
FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "REPLACE", "MERGE", "CALL",
    "EXEC", "EXECUTE", "LOAD_FILE", "OUTFILE", "DUMPFILE",
    "SET", "LOCK", "UNLOCK",
}

CANNOT_ANSWER_SENTINEL = "CANNOT_ANSWER"


def _strip_comments(sql: str) -> str:
    """
    Removes -- line comments and /* */ block comments before validation.
    Necessary because a comment can hide a second statement or a forbidden
    keyword from a naive check (e.g. "SELECT 1; -- DROP TABLE users" would
    pass a check that only scans for keywords outside of understanding
    comment syntax, if comments aren't stripped first).
    """
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


def _contains_forbidden_keyword(sql: str) -> str | None:
    """Returns the first forbidden keyword found as a whole word, or None."""
    upper = sql.upper()
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper):
            return keyword
    return None


def _is_single_statement(sql: str) -> bool:
    """
    Rejects multiple statements. Allows exactly one trailing semicolon
    (or none), rejects any semicolon followed by more non-whitespace
    content -- that's statement #2, regardless of what it contains.
    """
    trimmed = sql.strip()
    if trimmed.endswith(";"):
        trimmed = trimmed[:-1]
    return ";" not in trimmed


def validate_sql(sql: str, allowed_tables: list[str] | None = None) -> dict:
    """
    Args:
        sql: raw SQL string from sql_generator.generate_sql()
        allowed_tables: optional list of table names the query is permitted
                          to reference (typically ranked_context["table_names"]
                          from the same generation call). Defense-in-depth --
                          catches cases where the model referenced a table
                          outside the schema context it was given, which
                          would otherwise only surface as a confusing DB
                          error at execution time.

    Returns:
        {
          "valid": bool,
          "reason": str or None,   # human-readable rejection reason
          "cleaned_sql": str        # comment-stripped, trimmed SQL
                                      -- only meaningful if valid=True
        }
    """
    if sql.strip() == CANNOT_ANSWER_SENTINEL:
        return {
            "valid": False,
            "reason": "Model indicated the question cannot be answered with the given schema.",
            "cleaned_sql": None,
        }

    cleaned = _strip_comments(sql).strip()

    if not cleaned:
        return {"valid": False, "reason": "Empty SQL after stripping comments.", "cleaned_sql": None}

    if not re.match(r"^\s*SELECT\b", cleaned, re.IGNORECASE):
        return {
            "valid": False,
            "reason": f"Query does not start with SELECT. Got: {cleaned[:60]!r}",
            "cleaned_sql": None,
        }

    if not _is_single_statement(cleaned):
        return {
            "valid": False,
            "reason": "Multiple SQL statements detected. Only a single SELECT is allowed.",
            "cleaned_sql": None,
        }

    forbidden = _contains_forbidden_keyword(cleaned)
    if forbidden:
        return {
            "valid": False,
            "reason": f"Forbidden keyword detected: {forbidden}",
            "cleaned_sql": None,
        }

    if allowed_tables is not None:
        referenced = _extract_referenced_tables(cleaned)
        unknown = referenced - set(allowed_tables)
        if unknown:
            return {
                "valid": False,
                "reason": f"Query references table(s) outside provided schema context: {unknown}",
                "cleaned_sql": None,
            }

    return {"valid": True, "reason": None, "cleaned_sql": cleaned}


def _extract_referenced_tables(sql: str) -> set[str]:
    """
    Best-effort extraction of table names following FROM/JOIN. Not a full
    SQL parser -- this is a defense-in-depth heuristic, not the primary
    guardrail. Aliases (e.g. "FROM customer c") are handled; subqueries
    and CTEs are not specifically parsed but their inner FROM/JOIN clauses
    will still be picked up by the same regex.
    """
    pattern = r"\b(?:FROM|JOIN)\s+([`\"]?)(\w+)\1"
    matches = re.findall(pattern, sql, re.IGNORECASE)
    return {name for _, name in matches}


if __name__ == "__main__":
    test_cases = [
        ("SELECT * FROM customer;", ["customer"]),
        ("SELECT * FROM customer; DROP TABLE customer;", ["customer"]),
        ("DELETE FROM customer WHERE customer_id = 1;", ["customer"]),
        ("SELECT * FROM customer -- comment\n", ["customer"]),
        ("SELECT * FROM staff", ["customer"]),  # table not in allowed list
        ("CANNOT_ANSWER", None),
        ("SELECT c.customer_id FROM customer c JOIN rental r ON c.customer_id = r.customer_id", ["customer", "rental"]),
    ]

    for sql, allowed in test_cases:
        result = validate_sql(sql, allowed_tables=allowed)
        print(f"SQL: {sql!r}")
        print(f"  allowed_tables: {allowed}")
        print(f"  -> {result}")
        print()