from src.rag.hybrid_retriever import retrieve_hybrid
from src.rag.reranker import rerank_documents


question = "What is customer_unique_id?"


# 1. Hybrid candidate retrieval
documents = retrieve_hybrid(
    question=question,
    vector_k=10,
    bm25_k=10
)


# 2. Cross-encoder reranking
reranked_documents = rerank_documents(
    question=question,
    documents=documents,
    top_k=3,
    min_score=0.0
)


print("\nQuestion:")
print(question)

print("\nHybrid candidates:")
print(len(documents))

print("\n===== FINAL RERANKED RESULTS =====")


for rank, document in enumerate(
    reranked_documents,
    start=1
):

    print(f"\nRank {rank}")

    print(
        "Page:",
        document["metadata"].get("page")
    )

    print(
        "Vector Distance:",
        document.get("distance")
    )

    print(
        "BM25 Score:",
        document.get("bm25_score")
    )

    print(
        "Rerank Score:",
        round(
            document["rerank_score"],
            4
        )
    )

    print("Text:")
    print(document["text"][:300])