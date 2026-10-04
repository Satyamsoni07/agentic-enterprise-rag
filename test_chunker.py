from src.ingestion.text_loader import load_text_file
from src.ingestion.chunker import chunk_text


document = load_text_file(
    "data/raw/company_report.txt"
)

chunks = chunk_text(
    document,
    chunk_size=200,
    chunk_overlap=50
)

print("Total chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ---")
    print(chunk)