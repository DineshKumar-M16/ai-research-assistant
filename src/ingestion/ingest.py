import hashlib
from pathlib import Path

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import split_text
from src.embeddings.embedder import create_embeddings
from src.retrieval.vector_store import add_documents

PDF_PATH = "data/documents/sample.pdf"


def ingest_pdf(pdf_path: str, filename: str | None = None):
    path = Path(pdf_path)
    filename = filename or path.name
    print(f"Loading {filename}...")

    # Step 1: Extract text
    text = load_pdf(pdf_path)

    print(f"Characters extracted: {len(text)}")

    # Step 2: Split text into chunks
    chunks = split_text(text)

    print(f"Number of chunks: {len(chunks)}")

    if not chunks:
        print("No text chunks found.")
        return

    # Step 3: Create embeddings
    print("Creating embeddings...")

    embeddings = create_embeddings(chunks)

    print(f"Created {len(embeddings)} embeddings.")

    # Step 4: Create unique IDs
    digest = hashlib.sha256(filename.encode("utf-8") + path.read_bytes()).hexdigest()
    ids = [f"pdf_{digest}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"filename": filename} for _ in chunks]

    # Step 5: Store everything in ChromaDB
    print("Storing documents in ChromaDB...")

    add_documents(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    print("Ingestion completed successfully!")
    return {"filename": filename, "characters": len(text), "chunks": len(chunks)}


if __name__ == "__main__":
    ingest_pdf(PDF_PATH)
import hashlib
from pathlib import Path
