import os

from google import genai

from src.retrieval.search import search

MODEL_NAME = "gemini-3.8-flash"

SYSTEM_INSTRUCTIONS = """You are an AI research assistant.

Answer using the provided document context.
Do not invent facts.
If the answer is not available in the documents, say:
"I don't know based on the provided documents."
Use conversation history for follow-up questions.
"""


def generate_answer(
    question: str,
    conversation_history=None
) -> str:
    """Retrieve relevant document context and generate a grounded answer with Gemini."""

    if not question or not question.strip():
        return "Please enter a question."

    results = search(question, n_results=3)
    documents = results["documents"][0]

    if not documents:
        return "No relevant documents were found. Upload and process a research PDF, then try again."

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "Gemini API key is missing. Set GEMINI_API_KEY in your environment "
            "or add it to your Streamlit Community Cloud secrets."
        )

    context = "\n\n".join(documents)
    history = "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in (conversation_history or [])
    )
    if not history:
        history = "No prior conversation."

    prompt = f"""{SYSTEM_INSTRUCTIONS}

Conversation history:
{history}

Relevant document context:
{context}

Current question:
{question}
"""

    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
    except Exception as exc:
        raise RuntimeError(
            "Gemini could not generate an answer. Check the GEMINI_API_KEY "
            "and your network connection, then try again."
        ) from exc

    return response.text or "Gemini returned an empty response. Please try again."
