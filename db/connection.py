import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


load_dotenv()


def get_database_url() -> str:
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    database = os.getenv("DB_NAME")
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    if not all([host, port, database, username, password]):
        raise ValueError("Database configuration is incomplete.")

    return (
        f"mysql+pymysql://{username}:{password}"
        f"@{host}:{port}/{database}"
    )


def get_engine() -> Engine:
    return create_engine(
        get_database_url(),
        pool_pre_ping=True,
    )


def test_connection() -> None:
    engine = get_engine()

    with engine.connect() as connection:
        result = connection.execute(text("SELECT COUNT(*) FROM actor"))
        print(f"Database connection successful: {result.scalar()}")


if __name__ == "__main__":
    test_connection()