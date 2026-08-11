"""
api/table_access.py

Role-based table access control. Sits between rank_schema_context() and
sql_generator.generate_sql() -- filters which tables a given user's
question is allowed to see, BEFORE generation happens.

Current policy: fully permissive. No table categories have been flagged
as sensitive for this schema yet. Both "standard" and "admin" roles get
unrestricted access by default. This module exists so that restricting
access later (e.g. once HR/payroll tables are identified) is a config
change to ROLE_TABLE_RESTRICTIONS below, not a pipeline redesign.

Do not skip wiring this in just because it's a no-op today -- the value
is having the enforcement point already in place in the request flow.
"""

from typing import Optional

# role -> set of table names that role is FORBIDDEN from querying.
# Empty set = no restrictions (current default for both roles).
# Example future entry, once sensitive tables are identified:
#   "standard": {"payroll", "employee_salary", "ssn_records"}
ROLE_TABLE_RESTRICTIONS: dict[str, set[str]] = {
    "standard": set(),
    "admin": set(),
}


def filter_ranked_context(ranked_context: dict, user_role: str) -> dict:
    """
    Args:
        ranked_context: output of rank_schema_context.rank_schema_context()
        user_role: the requesting user's role (e.g. "standard", "admin")

    Returns:
        A copy of ranked_context with any forbidden tables removed from
        table_names, ranked_tables, and join_paths that reference them.
        Includes an additional "access_filtered" list showing what (if
        anything) was removed, for logging/audit purposes.
    """
    forbidden = ROLE_TABLE_RESTRICTIONS.get(user_role, set())

    if not forbidden:
        return {**ranked_context, "access_filtered": []}

    removed = [t for t in ranked_context["table_names"] if t in forbidden]
    allowed_table_names = [t for t in ranked_context["table_names"] if t not in forbidden]

    filtered_ranked_tables = [
        t for t in ranked_context["ranked_tables"] if t["table_name"] not in forbidden
    ]

    # Drop any join path that references a removed table on either side.
    filtered_join_paths = [
        jp for jp in ranked_context["join_paths"]
        if not any(
            part.split(".")[0] in forbidden
            for part in jp.split(" -> ")
        )
    ]

    return {
        "table_names": allowed_table_names,
        "ranked_tables": filtered_ranked_tables,
        "join_paths": filtered_join_paths,
        "truncated": ranked_context.get("truncated", False),
        "access_filtered": removed,
    }


def has_sufficient_access(filtered_context: dict) -> bool:
    """
    After filtering, checks whether there's anything left to work with.
    If access filtering removed every relevant table, generation should
    not proceed -- there's no point sending an empty schema context to
    the LLM, and the user should get a clear "access denied" message
    instead of a confusing downstream failure.
    """
    return len(filtered_context["table_names"]) > 0


if __name__ == "__main__":
    # Sanity check with the current (fully permissive) policy, and a
    # simulated restrictive policy to confirm filtering logic works
    # correctly once real restrictions are added later.
    sample_context = {
        "table_names": ["customer", "rental", "payment"],
        "ranked_tables": [
            {"table_name": "customer", "score": 0.4, "hops": 0},
            {"table_name": "rental", "score": 0.35, "hops": 0},
            {"table_name": "payment", "score": 0.3, "hops": 0},
        ],
        "join_paths": [
            "rental.customer_id -> customer.customer_id",
            "payment.rental_id -> rental.rental_id",
        ],
        "truncated": False,
    }

    print("Standard role (current permissive policy):")
    result = filter_ranked_context(sample_context, "standard")
    print(f"  tables: {result['table_names']}, filtered: {result['access_filtered']}")
    print(f"  sufficient access: {has_sufficient_access(result)}")

    print("\nSimulated restrictive policy (payment forbidden for standard):")
    ROLE_TABLE_RESTRICTIONS["standard"] = {"payment"}
    result = filter_ranked_context(sample_context, "standard")
    print(f"  tables: {result['table_names']}, filtered: {result['access_filtered']}")
    print(f"  join paths: {result['join_paths']}")
    print(f"  sufficient access: {has_sufficient_access(result)}")