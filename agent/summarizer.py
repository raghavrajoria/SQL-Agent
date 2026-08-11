"""
agent/summarizer.py

Generates a short natural-language summary of a successful query result,
for display in the UI next to the raw row data.

Deliberately a separate module and a separate Groq call from
sql_generator.py -- this only runs once, after execute_node has already
succeeded, so it must never be able to affect SQL generation, validation,
or the retry loop. If this call fails, the caller should treat it as
non-fatal and fall back to showing rows without a summary.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "qwen/qwen3.6-27b"

SUMMARY_SYSTEM_PROMPT = """You summarize SQL query results in plain English for a non-technical business user.
Rules:
1. 1-3 sentences. No SQL, no markdown, no column names verbatim unless it reads naturally.
2. State the direct answer to the question first.
3. If there are multiple rows, describe the pattern or top results -- don't just say "see table below".
4. No preamble like "Based on the data" or "According to the results".
"""


def summarize_result(question: str, columns: list[str], rows: list[dict], max_rows_in_prompt: int = 20) -> str:
    """
    Args:
        question: the original natural-language question
        columns: column names from the executed query
        rows: result rows (list of dicts)
        max_rows_in_prompt: cap on rows sent to the LLM, to keep this cheap
            regardless of MAX_ROWS_RETURNED upstream

    Returns:
        A short plain-text summary string.
    """
    sample_rows = rows[:max_rows_in_prompt]
    rows_text = "\n".join(str(r) for r in sample_rows)
    truncation_note = (
        f", showing first {max_rows_in_prompt}" if len(rows) > max_rows_in_prompt else ""
    )

    user_prompt = f"""QUESTION:
{question}

COLUMNS: {columns}

RESULT ROWS ({len(rows)} total{truncation_note}):
{rows_text}

Summary:"""

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        reasoning_effort="none",
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()