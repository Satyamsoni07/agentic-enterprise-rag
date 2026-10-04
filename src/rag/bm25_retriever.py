import re

from rank_bm25 import BM25Okapi

from src.vectorstore.chroma_store import get_all_documents


def tokenize(text: str) -> list[str]:

    # Convert text to lowercase
    text = text.lower()

    # Extract words and identifiers
    tokens = re.findall(
        r"\b\w+\b",
        text
    )

    return tokens


def retrieve_bm25(
    question: str,
    top_k: int = 10
) -> list[dict]:

    # 1. Get all stored chunks from Chroma
    results = get_all_documents()

    documents = results["documents"]
    metadatas = results["metadatas"]

    if not documents:
        return []

    # 2. Tokenize every document
    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    # 3. Build BM25 index
    bm25 = BM25Okapi(
        tokenized_documents
    )

    # 4. Tokenize the user's question
    tokenized_question = tokenize(
        question
    )

    # 5. Calculate BM25 scores
    scores = bm25.get_scores(
        tokenized_question
    )

    # 6. Combine text, metadata and score
    scored_documents = []

    for document, metadata, score in zip(
        documents,
        metadatas,
        scores
    ):
        scored_documents.append(
            {
                "text": document,
                "metadata": metadata,
                "bm25_score": float(score)
            }
        )

    # 7. Highest BM25 score = better match
    ranked_documents = sorted(
        scored_documents,
        key=lambda document: document["bm25_score"],
        reverse=True
    )

    return ranked_documents[:top_k]