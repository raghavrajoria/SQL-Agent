"""
agent/sql_generator.py

Takes the ranked schema context (rank_schema_context.py output) plus the
user's question, builds a prompt, and calls Groq (qwen/qwen3.6-27b) to
generate SQL.

This module ONLY generates SQL text. It does not validate or execute it --
that's the next two pieces in the pipeline (SQL validation, then the
execute + self-correction loop). Keeping generation, validation, and
execution as separate steps matters: it's the only way the self-correction
loop can cleanly re-invoke *just* generation with error feedback, without
re-running retrieval or re-building schema context every retry.
"""

import os
import re
from groq import Groq
from dotenv import load_dotenv

from schema_catalog.expand_schema import load_schema_catalog

load_dotenv()  # loads .env into os.environ -- must happen before Groq() is constructed

MODEL_NAME = "qwen/qwen3.6-27b"

# Columns that must never be sent to the LLM, regardless of which table
# ranks into context. This is enforced here, not just at description-
# generation time (generate_descriptions.py) -- a column can be schema-
# irrelevant for descriptions but still show up in a live SQL-generation
# prompt if its table gets selected. Extend this list as new sensitive
# columns are found; do not rely on remembering to check per-table.
SENSITIVE_COLUMN_NAMES = {"password", "ssn", "credit_card", "api_key", "secret"}


def build_schema_context_string(ranked_context: dict, schema_catalog: dict) -> str:
    """
    Turns ranked_context (table names) + schema_catalog (full column detail)
    into the raw schema text block for the prompt.

    Deliberately uses raw column name/type/PK/FK info, not the semantic
    descriptions -- once a table is selected, the LLM needs precise,
    literal schema facts to generate correct SQL, not a paraphrased summary.
    """
    lines = []
    tables = schema_catalog  # already unwrapped to {table_name: {...}}

    for table_name in ranked_context["table_names"]:
        table_info = tables.get(table_name)
        if not table_info:
            continue

        pk = set(table_info.get("primary_key", []))
        fk_cols = {
            fk["columns"][0]
            for fk in table_info.get("foreign_keys", [])
            if fk.get("columns")
        }

        col_lines = []
        for col in table_info.get("columns", []):
            name = col["name"]
            if name.lower() in SENSITIVE_COLUMN_NAMES:
                continue  # never expose sensitive columns to the LLM

            tags = []
            if name in pk:
                tags.append("PK")
            if name in fk_cols:
                tags.append("FK")
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            nullable = "NULL" if col.get("nullable") else "NOT NULL"

            col_lines.append(f"    {name} {col['type']} {nullable}{tag_str}")

        lines.append(f"TABLE {table_name} (")
        lines.extend(col_lines)
        lines.append(")")
        lines.append("")

    if ranked_context["join_paths"]:
        lines.append("KNOWN JOIN RELATIONSHIPS:")
        for jp in ranked_context["join_paths"]:
            lines.append(f"  {jp}")

    return "\n".join(lines)


SYSTEM_PROMPT = """You are an expert MySQL SQL generator.

RULES (follow exactly):
1. Generate ONLY a single MySQL SELECT statement. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, or any statement that modifies data or schema.
2. Use ONLY the tables and columns provided in the schema context below. Never invent table or column names.
3. Use the KNOWN JOIN RELATIONSHIPS provided to construct joins. Do not guess join conditions not listed there.
4. Return ONLY the raw SQL query. No explanation, no markdown code fences, no commentary, no reasoning text.
5. If the question cannot be answered with the given schema context, return exactly: CANNOT_ANSWER
"""


def _extract_sql(raw_text: str) -> str:
    """
    Strips markdown fences or stray text if the model doesn't follow rule 4
    exactly. Defensive parsing -- do not assume raw_text is already clean.
    """
    text = raw_text.strip()

    # Strip ```sql ... ``` or ``` ... ``` fences if present
    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    return text.strip()


def generate_sql(question: str, ranked_context: dict, error_context: dict | None = None) -> dict:
    """
    Args:
        question: the user's natural language question
        ranked_context: output of rank_schema_context.rank_schema_context()
        error_context: optional, used for retries after a failed attempt.
                         {"previous_sql": "...", "error_message": "..."}
                         When provided, the prompt includes the failed query
                         and the exact error, asking the model to fix it --
                         this is what the self-correction loop uses.

    Returns:
        {
          "sql": "SELECT ...",         # or "CANNOT_ANSWER"
          "raw_response": "...",        # unmodified model output, for debugging
          "schema_context_used": "..."  # the exact schema text sent, for logging
        }
    """
    schema_catalog = load_schema_catalog()
    schema_context_str = build_schema_context_string(ranked_context, schema_catalog)

    retry_block = ""
    if error_context:
        retry_block = f"""

YOUR PREVIOUS ATTEMPT FAILED. Fix it.
Previous SQL:
{error_context['previous_sql']}

Database error:
{error_context['error_message']}

Generate a corrected query that fixes this specific error. Follow all the same rules."""

    user_prompt = f"""SCHEMA CONTEXT:

{schema_context_str}

QUESTION:
{question}{retry_block}

SQL:"""

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        reasoning_effort="none",  # avoids raw reasoning text leaking into
                                    # output, same fix applied in
                                    # generate_descriptions.py
        temperature=0,             # deterministic SQL generation --
                                    # no creative variance wanted here
    )

    raw_text = response.choices[0].message.content
    sql = _extract_sql(raw_text)

    return {
        "sql": sql,
        "raw_response": raw_text,
        "schema_context_used": schema_context_str,
    }


if __name__ == "__main__":
    from schema_catalog.expand_schema import build_fk_graph
    from schema_catalog.rank_schema_context import rank_schema_context
    from schema_catalog.retrieve_schema import retrieve_schema

    schema_catalog = load_schema_catalog()
    fk_graph = build_fk_graph(schema_catalog)

    test_questions = [
        "Which customers rented the most movies?",
        "Which actors appeared in the most movies?",
        "Which films generated the most rental revenue?",
    ]

    for question in test_questions:
        retrieval_results = retrieve_schema(question, top_k=3)
        ranked_context = rank_schema_context(retrieval_results, fk_graph)
        result = generate_sql(question, ranked_context)

        print(f"\nQ: {question}")
        print(f"  tables used: {ranked_context['table_names']}")
        print(f"  SQL:\n{result['sql']}")