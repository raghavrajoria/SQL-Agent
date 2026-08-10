import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


DESCRIPTIONS_FILE = (
    Path(__file__).parent / "descriptions.json"
)

VECTOR_STORE_DIR = (
    Path(__file__).parent.parent / "vector_store" / "chroma"
)

COLLECTION_NAME = "schema_descriptions"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_descriptions() -> dict:
    with DESCRIPTIONS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_index() -> None:
    descriptions = load_descriptions()

    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(
        "Embedding dimension:",
        model.get_embedding_dimension(),
    )

    client = chromadb.PersistentClient(
        path=str(VECTOR_STORE_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    table_names = list(descriptions.keys())
    documents = [
        descriptions[table_name]
        for table_name in table_names
    ]

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
    ).tolist()

    collection.upsert(
        ids=table_names,
        documents=documents,
        embeddings=embeddings,
        metadatas=[
            {"table_name": table_name}
            for table_name in table_names
        ],
    )

    print("Vector index built successfully.")
    print("Documents:", collection.count())
    print("Collection:", COLLECTION_NAME)
    print("Storage:", VECTOR_STORE_DIR)


if __name__ == "__main__":
    build_index()