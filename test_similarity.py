from src.embeddings.ollama_embeddings import generate_embedding


def cosine_similarity(vector_a, vector_b):

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = sum(a * a for a in vector_a) ** 0.5
    magnitude_b = sum(b * b for b in vector_b) ** 0.5

    return dot_product / (magnitude_a * magnitude_b)


sentence_a = "The company generated revenue of $850 million."

sentence_b = "The business earned $850 million from sales."

sentence_c = "Cybersecurity is a major risk for the company."


embedding_a = generate_embedding(sentence_a)
embedding_b = generate_embedding(sentence_b)
embedding_c = generate_embedding(sentence_c)


similarity_ab = cosine_similarity(embedding_a,embedding_b)

similarity_ac = cosine_similarity(embedding_a,embedding_c)


print("A ↔ B similarity:", similarity_ab)
print("A ↔ C similarity:", similarity_ac)