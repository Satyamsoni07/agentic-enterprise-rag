from src.embeddings.ollama_embeddings import generate_embedding


text = "The company generated revenue of $850 million."

embedding = generate_embedding(text)

print("Original text:")
print(text)

print("\nEmbedding:")
print(embedding)

print("\nVector dimension:")
print(len(embedding))