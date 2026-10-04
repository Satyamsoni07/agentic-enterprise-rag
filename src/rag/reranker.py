from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

reranker = CrossEncoder(RERANKER_MODEL)


def rerank_documents(
    question: str,
    documents: list[dict],
    top_k: int = 3,
    min_score: float | None = None
) -> list[dict]:

    if not documents:
        return []

    pairs = [
        [question, document["text"]]
        for document in documents
    ]

    scores = reranker.predict(pairs)

    for document, score in zip(documents, scores):
        document["rerank_score"] = float(score)

    ranked_documents = sorted(
        documents,
        key=lambda document: document["rerank_score"],
        reverse=True
    )

    # Remove weak evidence if a threshold is provided
    if min_score is not None:
        ranked_documents = [
            document
            for document in ranked_documents
            if document["rerank_score"] >= min_score
        ]

    return ranked_documents[:top_k]