from src.rag.retriever import retrieve_documents
from src.rag.reranker import rerank_documents


question = "Why was a star schema chosen for the project?"


documents = retrieve_documents(question=question,top_k=10,max_distance=None)


reranked_documents = rerank_documents(question=question,documents=documents,top_k=3)


print("\nQuestion:")
print(question)

print("\n===== RERANKED RESULTS =====")

for rank, document in enumerate(reranked_documents,start=1):

    print(f"\nRank {rank}")

    print("Page:",document["metadata"].get("page"))

    print("Vector Distance:",round(document["distance"], 4))

    print("Rerank Score:",round(document["rerank_score"], 4))

    print("Text:")
    
    print(document["text"][:300])