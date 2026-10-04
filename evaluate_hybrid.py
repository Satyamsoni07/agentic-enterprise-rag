import json

from src.rag.hybrid_retriever import retrieve_hybrid
from src.rag.reranker import rerank_documents


with open(
    "data/evaluation/retrieval_eval.json",
    "r",
    encoding="utf-8"
) as file:
    evaluation_data = json.load(file)


successful_questions = 0


for item in evaluation_data:

    question = item["question"]
    expected_pages = item["expected_pages"]

    # 1. Retrieve candidates using BOTH
    # vector search and BM25
    documents = retrieve_hybrid(
        question=question,
        vector_k=10,
        bm25_k=10
    )

    # 2. Rerank the combined candidates
    reranked_documents = rerank_documents(
        question=question,
        documents=documents,
        top_k=1,
        min_score=None
    )

    # 3. Get final Rank-1 page
    top_page = (
        reranked_documents[0]["metadata"].get("page")
        if reranked_documents
        else None
    )

    # 4. Check whether Rank-1 is valid evidence
    success = top_page in expected_pages
    if not success:

        print("\n*** FAILED QUERY ***")
        print("Question:", question)
        print("Expected pages:", expected_pages)

        print("\nTop reranked results:")

        diagnostic_documents = rerank_documents(
            question=question,
            documents=documents,
            top_k=5,
            min_score=None
        )

        for rank, document in enumerate(
            diagnostic_documents,
            start=1
        ):
            print(
                f"Rank {rank} | "
                f"Page: {document['metadata'].get('page')} | "
                f"Score: {document['rerank_score']:.4f}"
            )

            print(
                document["text"][:250]
            )

            print()

    if success:
        successful_questions += 1

    print("\n----------------------------------")
    print("Question:")
    print(question)

    print("Expected pages:", expected_pages)
    print("Hybrid + Reranker Rank 1:", top_page)
    print("Found:", success)


total_questions = len(evaluation_data)

hit_rate = (
    successful_questions / total_questions
)


print("\n==================================")

print(
    f"Hybrid + Reranker Hit Rate@1: "
    f"{hit_rate:.2%}"
)

print(
    f"Successful: "
    f"{successful_questions}/{total_questions}"
)

print("==================================")