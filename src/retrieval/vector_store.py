import chromadb

client = chromadb.PersistentClient(
    path="./data/chroma"
)

collection = client.get_or_create_collection(
    name="research_documents"
)


def add_documents(
    documents: list[str],
    embeddings,
    ids: list[str],
    metadatas: list[dict] | None = None,
):
    # Upsert makes reprocessing the same file safe while preserving other PDFs.
    collection.upsert(
        documents=documents,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=metadatas,
    )


def search_documents(
    query_embedding,
    n_results: int = 3
):
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )

    return results


def get_document_count():
    return collection.count()


if __name__ == "__main__":
    print("ChromaDB vector store is ready.")
    print("Collection:", collection.name)
    print("Documents:", get_document_count())
