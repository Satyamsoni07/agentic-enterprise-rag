import json

from src.rag.retriever import retrieve_documents
from src.rag.reranker import rerank_documents


with open(
    "data/evaluation/retrieval_eval.json",
    "r",
    encoding="utf-8"
) as file:
    evaluation_data = json.load(file)


successful_before = 0
successful_after = 0


for item in evaluation_data:

    question = item["question"]
    expected_pages = item["expected_pages"]

    # -----------------------------
    # VECTOR RETRIEVAL
    # -----------------------------

    documents = retrieve_documents(
        question=question,
        top_k=10,
        max_distance=None
    )

    vector_top_page = (
        documents[0]["metadata"].get("page")
        if documents
        else None
    )

    vector_success = (
        vector_top_page in expected_pages
    )

    if vector_success:
        successful_before += 1

    # -----------------------------
    # RERANKING
    # -----------------------------

    reranked_documents = rerank_documents(
        question=question,
        documents=documents,
        top_k=3
    )

    reranked_top_page = (
        reranked_documents[0]["metadata"].get("page")
        if reranked_documents
        else None
    )

    rerank_success = (
        reranked_top_page in expected_pages
    )

    if rerank_success:
        successful_after += 1

    # -----------------------------
    # PRINT QUESTION RESULT
    # -----------------------------

    print("\n----------------------------------")
    print("Question:")
    print(question)

    print("Expected pages:", expected_pages)

    print(
        "Vector Rank 1:",
        vector_top_page,
        "→",
        vector_success
    )

    print(
        "Reranker Rank 1:",
        reranked_top_page,
        "→",
        rerank_success
    )


total_questions = len(evaluation_data)

vector_hit_rate = (
    successful_before / total_questions
)

reranker_hit_rate = (
    successful_after / total_questions
)


print("\n==================================")

print(
    f"Vector Search Hit Rate@1: "
    f"{vector_hit_rate:.2%}"
)

print(
    f"Reranker Hit Rate@1: "
    f"{reranker_hit_rate:.2%}"
)

print(
    f"Vector Search Successful: "
    f"{successful_before}/{total_questions}"
)

print(
    f"Reranker Successful: "
    f"{successful_after}/{total_questions}"
)

print("==================================")