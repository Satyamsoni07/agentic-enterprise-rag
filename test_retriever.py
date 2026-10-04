from src.rag.retriever import retrieve_documents


question = "What is the population of Japan?"

documents = retrieve_documents(
    question,
    top_k=3,
    max_distance=0.75
)


print("Question:")
print(question)

print("\nNumber of documents retrieved:", len(documents))


if not documents:
    print("\nNo relevant documents found.")

else:
    for i, document in enumerate(documents, start=1):

        print(f"\n--- Result {i} ---")

        print(
            "Distance:",
            round(document["distance"], 4)
        )

        print(
            "Page:",
            document["metadata"]["page"]
        )

        print("Text:")
        print(document["text"])