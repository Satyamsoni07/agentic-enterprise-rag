from src.embeddings.ollama_embeddings import generate_embedding
from src.vectorstore.chroma_store import search_chunks


def retrieve_documents(question: str,top_k: int = 3,max_distance: float | None = None) -> list[dict]:

    question_embedding = generate_embedding(question)

    results = search_chunks(query_embedding=question_embedding,top_k=top_k)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved_documents = []

    for document, metadata, distance in zip(documents,metadatas,distances):

        # Skip weak matches if a threshold is provided
        if (max_distance is not None and distance > max_distance):
            continue

        retrieved_documents.append(
            {
                "text": document,
                "metadata": metadata,
                "distance": distance
            }
        )

    return retrieved_documents