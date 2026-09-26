from ollama import chat

from src.retrieval.search import search

MODEL_NAME = "llama3.2"


def generate_answer(
    question: str,
    conversation_history=None
) -> str:
    """

    if not question or not question.strip():
        return "Please enter a question."
    Retrieve relevant documents and generate an answer
    using the conversation history.
    """

    results = search(
        question,
        n_results=3
    )

    documents = results["documents"][0]

    if not documents:
        return "No relevant documents were found. Upload and process a research PDF, then try again."

    context = "\n\n".join(documents)

    messages = [
        {
            "role": "system",
            "content": """
You are an AI research assistant.

Answer questions using the provided document context.

Rules:
- Use the document context as the main source of information.
- Do not invent facts.
- If the answer is not available in the documents, say:
  "I don't know based on the provided documents."
- Use conversation history to understand follow-up questions.
"""
        }
    ]

    if conversation_history:

        for message in conversation_history:

            messages.append(
                {
                    "role": message["role"],
                    "content": message["content"]
                }
            )

    messages.append(
        {
            "role": "user",
            "content": f"""
Relevant document context:

{context}

Current question:

{question}
"""
        }
    )

    try:
        response = chat(model=MODEL_NAME, messages=messages)
    except Exception as exc:
        raise RuntimeError("Ollama or the Llama 3.2 model is unavailable. Start Ollama and ensure llama3.2 is installed.") from exc

    return response.message.content
