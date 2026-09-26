import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.ingestion.ingest import ingest_pdf
from src.rag.generator import generate_answer

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖"
)

st.title("🤖 AI Research Assistant")

st.write(
    "Upload a research PDF and ask questions about it."
)

# -----------------------------
# Chat Memory
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# PDF Upload
# -----------------------------

uploaded_files = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} PDF file(s).")
    if st.button("Process PDF"):
        for uploaded_file in uploaded_files:
            st.markdown(f"**PDF:** {uploaded_file.name}")
            try:
                if not uploaded_file.getbuffer().nbytes:
                    st.error("The uploaded PDF is empty. Choose a PDF that contains pages and try again.")
                    continue
                safe_filename = Path(uploaded_file.name).name
                pdf_path = PROJECT_ROOT / "data" / "documents" / safe_filename
                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                with open(pdf_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())

                st.info("Extracting text...")
                result = ingest_pdf(str(pdf_path), filename=safe_filename)
                if not result or not result["chunks"]:
                    st.warning("No text chunks could be created from this PDF.")
                else:
                    st.write(f"Extracted characters: {result['characters']}")
                    st.write(f"Chunks: {result['chunks']}")
                    st.success(f"{result['filename']} processed and stored successfully!")
            except ValueError as exc:
                st.error(str(exc))
            except Exception:
                st.error(f"Processing failed for {uploaded_file.name}. Check that it is a readable PDF and try again.")
else:
    st.warning("Upload a PDF to process research documents.")

st.divider()

st.subheader("💬 Chat")

# -----------------------------
# Display previous messages
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

# -----------------------------
# Chat Input
# -----------------------------

question = st.chat_input(
    "Ask a question about your research..."
)

if question:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents..."
        ):

            try:
                answer = generate_answer(question, st.session_state.messages)
            except RuntimeError as exc:
                answer = str(exc)
                st.error(answer)
            except Exception:
                answer = "Something went wrong while answering. Please try again."
                st.error(answer)

        st.write(answer)

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
