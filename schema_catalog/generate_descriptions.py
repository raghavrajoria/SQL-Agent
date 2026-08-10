import json
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text
from groq import Groq

from db.connection import get_engine


load_dotenv()


SCHEMA_FILE = Path(__file__).parent / "schema_catalog.json"
DESCRIPTIONS_FILE = Path(__file__).parent / "descriptions.json"

MODEL_NAME = "qwen/qwen3.6-27b"
SAMPLE_ROW_COUNT = 5


def load_schema() -> dict:
    with SCHEMA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_sample_rows(
    table_name: str,
    table_schema: dict,
) -> list[dict]:
    engine = get_engine()

    sample_columns = [
        column["name"]
        for column in table_schema["columns"]
        if "BLOB" not in column["type"].upper()
        and "BINARY" not in column["type"].upper()
    ]

    if not sample_columns:
        return []

    columns = ", ".join(
        f"`{column}`"
        for column in sample_columns
    )

    query = text(
        f"SELECT {columns} "
        f"FROM `{table_name}` "
        f"LIMIT {SAMPLE_ROW_COUNT}"
    )

    with engine.connect() as connection:
        result = connection.execute(query)
        return [dict(row._mapping) for row in result]


def build_prompt(
    table_name: str,
    table_schema: dict,
    sample_rows: list[dict],
) -> str:
    columns = "\n".join(
        f"- {column['name']} ({column['type']})"
        for column in table_schema["columns"]
    )

    foreign_keys = "\n".join(
        f"- {', '.join(fk['columns'])} -> "
        f"{fk['referred_table']}."
        f"{', '.join(fk['referred_columns'])}"
        for fk in table_schema["foreign_keys"]
    )

    return f"""
You are documenting a relational database schema for a Text-to-SQL system.

Create a concise semantic description of the database table below.

The description should explain:
- what the table represents
- what kind of records it stores
- the meaning of the important columns
- important relationships to other tables

Do not invent information that cannot be inferred from the schema
or sample data.

Table:
{table_name}

Columns:
{columns}

Primary key:
{", ".join(table_schema["primary_key"])}

Foreign keys:
{foreign_keys if foreign_keys else "None"}

Sample rows:
{json.dumps(sample_rows, default=str, indent=2)}

Return ONLY the final table description.

Do not include:
- thinking or reasoning
- analysis
- step-by-step explanations
- drafts
- reviews
- headings
- commentary
- discussion of how you arrived at the description

Your entire response must consist only of the concise
semantic description of the table.
""".strip()


def generate_description(
    client: Groq,
    table_name: str,
    table_schema: dict,
    sample_rows: list[dict],
) -> str:
    prompt = build_prompt(
        table_name,
        table_schema,
        sample_rows,
    )

    response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    temperature=0,
    reasoning_effort="none",
    )

    return response.choices[0].message.content.strip()


def save_description(table_name: str, description: str) -> None:
    if DESCRIPTIONS_FILE.exists():
        with DESCRIPTIONS_FILE.open("r", encoding="utf-8") as file:
            descriptions = json.load(file)
    else:
        descriptions = {}

    descriptions[table_name] = description

    with DESCRIPTIONS_FILE.open("w", encoding="utf-8") as file:
        json.dump(descriptions, file, indent=2)


def generate_table_description(table_name: str) -> None:
    if DESCRIPTIONS_FILE.exists():
            with DESCRIPTIONS_FILE.open("r", encoding="utf-8") as file:
                descriptions = json.load(file)

            if table_name in descriptions:
                print(f"Description already exists for table: {table_name}")
                print(descriptions[table_name])
                return
            
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured.")

    schema = load_schema()

    if table_name not in schema["tables"]:
        raise ValueError(
            f"Table '{table_name}' was not found in schema catalog."
        )

    table_schema = schema["tables"][table_name]
    sample_rows = get_sample_rows(table_name, table_schema)

    client = Groq(api_key=api_key)

    description = generate_description(
        client,
        table_name,
        table_schema,
        sample_rows,
    )

    save_description(table_name, description)

    print(f"Description generated for table: {table_name}")
    print()
    print(description)


if __name__ == "__main__":
    generate_table_description("store")