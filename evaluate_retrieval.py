import json

from src.rag.retriever import retrieve_documents


# Load our evaluation questions
with open("data/evaluation/retrieval_eval.json",
          "r",
          encoding="utf-8") as file:
    evaluation_data = json.load(file)


def evaluate_recall(k: int):

    successful_questions = 0

    for item in evaluation_data:

        question = item["question"]
        expected_pages = item["expected_pages"]

        # Retrieve top K documents.
        # No distance threshold yet because we first
        # want to measure raw retrieval quality.
        documents = retrieve_documents(
            question=question,
            top_k=k,
            max_distance=None
        )

        retrieved_pages = [
            document["metadata"].get("page")
            for document in documents
        ]

        found = any(page in retrieved_pages for page in expected_pages )

        if found:
            successful_questions += 1

        print("\nQuestion:")
        print(question)

        print("Expected pages:", expected_pages)
        print("Retrieved pages:", retrieved_pages)
        print("Found:", found)

    recall = successful_questions / len(evaluation_data)

    print("\n----------------------------")
    print(f"Recall@{k}: {recall:.2%}")
    print(
        f"Successful: "
        f"{successful_questions}/{len(evaluation_data)}"
    )
    print("----------------------------")

    return recall


print("\n===== RECALL@1 =====")
evaluate_recall(1)

print("\n===== RECALL@3 =====")
evaluate_recall(3)

print("\n===== RECALL@5 =====")
evaluate_recall(5)