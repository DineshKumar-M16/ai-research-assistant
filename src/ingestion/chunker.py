from langchain_text_splitters import RecursiveCharacterTextSplitter

if __package__:
    from src.ingestion.pdf_loader import load_pdf
else:
    from pdf_loader import load_pdf


def split_text(text: str) -> list[str]:
    """
    Split extracted text into smaller overlapping chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )

    return splitter.split_text(text)


if __name__ == "__main__":
    pdf_path = "data/documents/sample.pdf"

    text = load_pdf(pdf_path)

    print(f"Characters extracted: {len(text)}")

    chunks = split_text(text)

    print(f"Number of chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk[:500])