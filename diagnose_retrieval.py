import json

from src.rag.retriever import retrieve_documents


with open(
    "data/evaluation/retrieval_eval.json",
    "r",
    encoding="utf-8"
) as file:
    evaluation_data = json.load(file)


print("===== FAILED RECALL@3 QUERIES =====")

failed_count = 0

for item in evaluation_data:

    question = item["question"]
    expected_page = item["expected_page"]

    documents = retrieve_documents(
        question=question,
        top_k=3,
        max_distance=None
    )

    retrieved_pages = [
        document["metadata"].get("page")
        for document in documents
    ]

    # We only want to inspect failures
    if expected_page not in retrieved_pages:

        failed_count += 1

        print("\n----------------------------------")
        print("Question:")
        print(question)

        print("\nExpected page:", expected_page)

        print("\nRetrieved results:")

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
                "Distance:",
                round(document["distance"], 4)
            )
            print("Text:")
            print(document["text"])


print("\n==================================")
print("Total failed questions:", failed_count)