from src.ingestion.text_loader import load_text_file
from src.ingestion.recursive_chunker import recursive_chunk_text
from src.embeddings.ollama_embeddings import generate_embedding
from src.vectorstore.chroma_store import add_chunks


# 1. Load
document = load_text_file("data/raw/company_report.txt")


# 2. Chunk
chunks = recursive_chunk_text(
    document,
    chunk_size=200,
    chunk_overlap=50
)


# 3. Embed
embeddings = [
    generate_embedding(chunk)
    for chunk in chunks
]


# 4. Store
add_chunks(
    chunks=chunks,
    embeddings=embeddings,
    source="company_report.txt"
)


print("Document indexed successfully.")
print("Total chunks stored:", len(chunks))