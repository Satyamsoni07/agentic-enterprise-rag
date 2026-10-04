from src.rag.bm25_retriever import retrieve_bm25


question = "What is customer_unique_id?"


documents = retrieve_bm25(
    question=question,
    top_k=5
)


print("\nQuestion:")
print(question)

print("\n===== BM25 RESULTS =====")


for rank, document in enumerate(
    documents,
    start=1
):

    print(f"\nRank {rank}")

    print(
        "Page:",
        document["metadata"].get("page")
    )

    print(
        "BM25 Score:",
        round(document["bm25_score"], 4)
    )

    print("Text:")
    print(document["text"][:300])