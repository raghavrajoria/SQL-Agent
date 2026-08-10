"""
schema_catalog/expand_schema.py

Deterministic foreign-key graph expansion.

Semantic retrieval (retrieve_schema.py) finds tables that are *semantically*
relevant to a question. It does not guarantee *structural* completeness --
e.g. a "customer rentals" question may retrieve `rental` and `payment` but
miss `customer`, even though `customer` is required to join and answer the
question at all.

This module fixes that gap: given a set of candidate tables from semantic
retrieval, it walks the FK graph outward (1-2 hops) and pulls in any table
directly connected to a candidate, so join paths are structurally complete
before the schema context ever reaches the LLM.

Actual schema_catalog.json shape:

{
  "database": ...,
  "tables": {
    "customer": {
      "columns": [...],
      "primary_key": ["customer_id"],
      "foreign_keys": [
        {
          "columns": ["address_id"],
          "referred_table": "address",
          "referred_columns": ["address_id"]
        }
      ]
    },
    ...
  },
  "views": {...}
}

load_schema_catalog() unwraps the "tables" key so downstream functions
just work with {table_name: {...}}.
"""

import json
from pathlib import Path
from collections import deque

SCHEMA_CATALOG_PATH = Path(__file__).parent / "schema_catalog.json"


def load_schema_catalog(path: Path = SCHEMA_CATALOG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        full = json.load(f)
    # Actual file shape: {"database": ..., "tables": {table_name: {...}}, "views": {...}}
    # We only need the table-level dict for FK graph building.
    return full["tables"]


def build_fk_graph(schema_catalog: dict) -> dict:
    """
    Builds a bidirectional adjacency graph from FK relationships.

    Returns:
        {
          "rental": {
              "customer": {"via": "rental.customer_id -> customer.customer_id"},
              "inventory": {"via": "rental.inventory_id -> inventory.inventory_id"},
              ...
          },
          ...
        }

    Bidirectional because a question might semantically hit the "parent"
    side (e.g. "customer") and still need the "child" side (e.g. "rental")
    pulled in, or vice versa.
    """
    graph = {table: {} for table in schema_catalog.keys()}

    for table_name, table_info in schema_catalog.items():
        fks = table_info.get("foreign_keys", [])
        for fk in fks:
            referred_table = fk.get("referred_table")
            if not referred_table or referred_table not in schema_catalog:
                continue

            constrained_cols = fk.get("columns", [])
            referred_cols = fk.get("referred_columns", [])
            via_desc = (
                f"{table_name}.{','.join(constrained_cols)} -> "
                f"{referred_table}.{','.join(referred_cols)}"
            )

            # forward edge: child -> parent
            graph[table_name][referred_table] = {"via": via_desc}
            # reverse edge: parent -> child (bidirectional)
            graph.setdefault(referred_table, {})
            graph[referred_table][table_name] = {"via": via_desc}

    return graph


def expand_tables(
    candidate_tables: list[str],
    fk_graph: dict,
    max_hops: int = 1,
) -> dict:
    """
    BFS outward from candidate tables through the FK graph.

    Args:
        candidate_tables: table names returned by semantic retrieval
        fk_graph: output of build_fk_graph()
        max_hops: how many FK hops to expand outward. 1 is usually enough
                   for typical join questions; 2 risks pulling in too much
                   noise on a densely connected schema like Sakila.

    Returns:
        {
          "tables": ["rental", "customer", "inventory", ...],
          "join_paths": [
              "rental.customer_id -> customer.customer_id",
              "rental.inventory_id -> inventory.inventory_id",
              ...
          ]
        }
    """
    visited = set(candidate_tables)
    join_paths = []

    queue = deque([(t, 0) for t in candidate_tables])

    while queue:
        current, depth = queue.popleft()
        if depth >= max_hops:
            continue

        for neighbor, edge_info in fk_graph.get(current, {}).items():
            join_paths.append(edge_info["via"])
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return {
        "tables": sorted(visited),
        "join_paths": sorted(set(join_paths)),
    }


if __name__ == "__main__":
    # Quick manual test against your three checkpoint questions.
    # Replace these candidate lists with actual output from retrieve_schema.py
    # once you wire the two together.

    schema_catalog = load_schema_catalog()
    fk_graph = build_fk_graph(schema_catalog)

    test_cases = {
        "Which customers rented the most movies?": ["rental", "payment"],
        "Which actors appeared in the most movies?": ["film_actor", "actor"],
        "Which films generated the most rental revenue?": ["payment", "film"],
    }

    for question, candidates in test_cases.items():
        result = expand_tables(candidates, fk_graph, max_hops=1)
        print(f"\nQ: {question}")
        print(f"  candidates: {candidates}")
        print(f"  expanded tables: {result['tables']}")
        print(f"  join paths:")
        for jp in result["join_paths"]:
            print(f"    {jp}")