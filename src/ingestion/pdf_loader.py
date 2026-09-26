from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(path))

    if not reader.pages:
        raise ValueError("The PDF is empty and contains no pages.")

    text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        text += f"\n--- Page {page_number} ---\n"
        text += page_text

    if not text.strip():
        raise ValueError("No extractable text was found in the PDF.")

    return text


if __name__ == "__main__":
    pdf_path = "data/documents/sample.pdf"

    text = load_pdf(pdf_path)

    print("\nPDF loaded successfully!")
    print(f"Characters extracted: {len(text)}")

    print("\n--- Extracted Text ---")
    print(text[:2000])
