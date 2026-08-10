import json
from pathlib import Path

from sqlalchemy import inspect

from db.connection import get_engine


OUTPUT_FILE = Path(__file__).parent / "schema_catalog.json"


def extract_schema() -> dict:
    engine = get_engine()
    inspector = inspect(engine)

    table_names = inspector.get_table_names()
    view_names = inspector.get_view_names()

    catalog = {
        "database": engine.url.database,
        "tables": {},
        "views": {},
    }

    for table_name in table_names:
        columns = inspector.get_columns(table_name)
        primary_key = inspector.get_pk_constraint(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)

        catalog["tables"][table_name] = {
            "columns": [
                {
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": column["nullable"],
                    "default": column["default"],
                }
                for column in columns
            ],
            "primary_key": primary_key.get("constrained_columns", []),
            "foreign_keys": [
                {
                    "columns": foreign_key.get("constrained_columns", []),
                    "referred_table": foreign_key.get("referred_table"),
                    "referred_columns": foreign_key.get(
                        "referred_columns", []
                    ),
                }
                for foreign_key in foreign_keys
            ],
        }

    for view_name in view_names:
        columns = inspector.get_columns(view_name)

        catalog["views"][view_name] = {
            "columns": [
                {
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": column["nullable"],
                }
                for column in columns
            ]
        }

    return catalog


def save_schema(catalog: dict) -> None:
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(catalog, file, indent=2, default=str)


if __name__ == "__main__":
    schema = extract_schema()
    save_schema(schema)

    print(f"Schema extracted successfully.")
    print(f"Tables: {len(schema['tables'])}")
    print(f"Views: {len(schema['views'])}")
    print(f"Output: {OUTPUT_FILE}")