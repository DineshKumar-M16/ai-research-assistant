from src.embeddings.embedder import create_embeddings
from src.retrieval.vector_store import search_documents


def search(query: str, n_results: int = 3):
    """
    Search ChromaDB for documents semantically related
    to the user's question.
    """

    if not query or not query.strip():
        raise ValueError("Please enter a question.")

    try:
        query_embedding = create_embeddings([query])[0]
    except Exception as exc:
        raise RuntimeError("Could not create a search embedding. Please try again.") from exc

    # Search ChromaDB
    try:
        results = search_documents(query_embedding, n_results=n_results)
    except Exception as exc:
        raise RuntimeError("Document retrieval failed. Check that documents are processed and the vector store is available.") from exc

    return results


if __name__ == "__main__":

    query = "What is artificial intelligence?"

    results = search(query, n_results=3)

    print("\nSearch results:")

    for i, document in enumerate(results["documents"][0], start=1):

        print(f"\n--- Result {i} ---")
        print(document)
