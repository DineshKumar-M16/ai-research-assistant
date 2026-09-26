"""Basic retrieval evaluation for the AI Research Assistant."""

from src.retrieval.search import search


TEST_CASES = [
    ("What are embeddings?", "embeddings"),
    ("What is retrieval augmented generation?", "retrieval"),
    ("What is text chunking?", "chunking"),
    ("What is semantic search?", "semantic search"),
    ("What are vector databases?", "vector databases"),
]


def evaluate_rag():
    passed = 0
    failed = 0

    print("=== RAG Retrieval Evaluation ===\n")

    for question, expected_topic in TEST_CASES:
        results = search(question, n_results=3)
        documents = results["documents"][0]
        retrieved_chunks = len(documents)
        retrieved_content = " ".join(documents).casefold()
        status = "PASS" if expected_topic.casefold() in retrieved_content else "FAIL"
        passed += status == "PASS"
        failed += status == "FAIL"

        print(f"Question: {question}")
        print(f"Expected topic: {expected_topic}")
        print(f"Retrieved chunks: {retrieved_chunks}")
        print(f"Status: {status}")
        print("------------------------------------------------------------\n")

    print("=== Evaluation Summary ===")
    print(f"Total questions: {len(TEST_CASES)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    evaluate_rag()
