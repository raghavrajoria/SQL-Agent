import chromadb
from sentence_transformers import SentenceTransformer


VECTOR_STORE_DIR = "vector_store/chroma"
COLLECTION_NAME = "schema_descriptions"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def get_collection():
    client = chromadb.PersistentClient(
        path=VECTOR_STORE_DIR
    )

    return client.get_collection(
        name=COLLECTION_NAME
    )


def retrieve_schema(
    question: str,
    top_k: int = 5,
) -> list[dict]:
    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    question_embedding = model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    collection = get_collection()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved = []

    for i in range(len(results["ids"][0])):
        retrieved.append(
            {
                "table_name": results["metadatas"][0][i]["table_name"],
                "description": results["documents"][0][i],
                "distance": results["distances"][0][i],
            }
        )

    return retrieved


if __name__ == "__main__":
    question = "Which customers rented the most movies?"

    results = retrieve_schema(question)

    print(f"Question: {question}")
    print()

    for result in results:
        print(
            f"Table: {result['table_name']}"
        )
        print(
            f"Distance: {result['distance']}"
        )
        print(
            f"Description: {result['description']}"
        )
        print("-" * 80)