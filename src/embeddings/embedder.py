from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

model = None


def create_embeddings(texts: list[str]):
    """
    Convert text chunks into numerical vectors.
    """
    global model
    if model is None:
        model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(texts)

    return embeddings


if __name__ == "__main__":
    texts = [
        "Artificial intelligence is a field of computer science.",
        "Machine learning allows computers to learn from data.",
        "Python is a popular programming language.",
    ]

    embeddings = create_embeddings(texts)

    print("Number of texts:", len(texts))
    print("Number of embeddings:", len(embeddings))
    print("Vector dimensions:", len(embeddings[0]))

    print("\nFirst embedding:")
    print(embeddings[0])
