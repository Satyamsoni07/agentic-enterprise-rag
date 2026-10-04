from src.rag.hybrid_retriever import retrieve_hybrid


question = "What is customer_unique_id?"


documents = retrieve_hybrid(
    question=question,
    vector_k=10,
    bm25_k=10
)


print("\nQuestion:")
print(question)

print("\nTotal unique hybrid candidates:")
print(len(documents))


print("\n===== HYBRID CANDIDATES =====")

for rank, document in enumerate(
    documents,
    start=1
):

    print(f"\nCandidate {rank}")

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

    print("Text:")
    print(document["text"][:200])