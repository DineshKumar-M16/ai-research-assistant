from src.retrieval.search import search


def search_documents_tool(question: str) -> str:
    """
    Search the research documents and return relevant information.
    """

    results = search(
        question,
        n_results=3
    )

    documents = results["documents"][0]

    if not documents:
        return "No relevant information was found."

    return "\n\n".join(documents)


if __name__ == "__main__":

    question = "What are embeddings?"

    result = search_documents_tool(question)

    print("\n--- Retrieved Information ---")
    print(result)