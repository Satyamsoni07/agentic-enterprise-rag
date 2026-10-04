import chromadb


client = chromadb.PersistentClient(path="data/chroma_db")


collection = client.get_or_create_collection(name="enterprise_documents")


def add_chunks(chunks, embeddings, source):

    ids = [
        f"{source}_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {"source": source}
        for _ in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

def search_chunks(query_embedding, top_k=3):

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return results

def add_documents(documents, embeddings):

    ids = []

    for i, document in enumerate(documents):

        source = document["metadata"]["source"]
        page = document["metadata"]["page"]

        safe_source = (
            source
            .replace("\\", "_")
            .replace("/", "_")
            .replace(" ", "_")
        )

        chunk_id = (f"{safe_source}_page_{page}_chunk_{i}")

        ids.append(chunk_id)

    texts = [document["text"]for document in documents]

    metadatas = [document["metadata"]for document in documents]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

def get_all_documents():

    results = collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    return results