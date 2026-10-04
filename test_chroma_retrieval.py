from src.embeddings.ollama_embeddings import generate_embedding
from src.vectorstore.chroma_store import search_chunks


question = "Why did you choose a star schema for this project?"

question_embedding = generate_embedding(question)

results = search_chunks(query_embedding=question_embedding,top_k=3)


documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]


for i, (document, metadata, distance) in enumerate(
    zip(documents, metadatas, distances),
    start=1
):

    print(f"\n--- Result {i} ---")

    print("Distance:", round(distance, 4))

    print(document)

    print("\nSource:")
    print(metadata["source"])

    print("Page:")
    print(metadata["page"])