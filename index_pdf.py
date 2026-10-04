from src.ingestion.pdf_loader import load_pdf
from src.ingestion.document_chunker import chunk_documents
from src.embeddings.ollama_embeddings import generate_embedding
from src.vectorstore.chroma_store import add_documents


# 1. Load PDF
pages = load_pdf("data/raw/sample_report.pdf")


# 2. Chunk PDF
documents = chunk_documents(pages)


# 3. Generate embedding for every chunk
embeddings = [generate_embedding(document["text"])for document in documents]


# 4. Store in Chroma
add_documents(documents=documents,embeddings=embeddings)


print("PDF indexed successfully.")
print("Total pages:", len(pages))
print("Total chunks stored:", len(documents))