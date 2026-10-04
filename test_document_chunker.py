from src.ingestion.pdf_loader import load_pdf
from src.ingestion.document_chunker import chunk_documents


pages = load_pdf("data/raw/sample_report.pdf")

chunks = chunk_documents(pages)


print("Total pages:", len(pages))
print("Total chunks:", len(chunks))


for chunk in chunks[:5]:

    print("\n--------------------")

    print("Metadata:")
    print(chunk["metadata"])

    print("\nChunk:")
    print(chunk["text"])