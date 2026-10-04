from src.rag.retriever import retrieve_documents
from src.rag.bm25_retriever import retrieve_bm25


def retrieve_hybrid(
    question: str,
    vector_k: int = 10,
    bm25_k: int = 10
) -> list[dict]:

    # 1. Semantic/vector retrieval
    vector_documents = retrieve_documents(
        question=question,
        top_k=vector_k,
        max_distance=None
    )

    # 2. Keyword/BM25 retrieval
    bm25_documents = retrieve_bm25(
        question=question,
        top_k=bm25_k
    )

    # 3. Merge both candidate lists
    combined_documents = (
        vector_documents
        + bm25_documents
    )

    # 4. Remove duplicate chunks
    unique_documents = []
    seen_chunks = set()

    for document in combined_documents:

        text = document["text"]

        if text not in seen_chunks:
            seen_chunks.add(text)
            unique_documents.append(document)

    return unique_documents