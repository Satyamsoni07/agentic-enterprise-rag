from src.ingestion.text_loader import load_text_file
from src.ingestion.recursive_chunker import recursive_chunk_text
from src.embeddings.ollama_embeddings import generate_embedding


def cosine_similarity(vector_a, vector_b):

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = sum(a * a for a in vector_a) ** 0.5
    magnitude_b = sum(b * b for b in vector_b) ** 0.5

    return dot_product / (magnitude_a * magnitude_b)


# 1. Load document
document = load_text_file("data/raw/company_report.txt")


# 2. Split document into chunks
chunks = recursive_chunk_text(document,chunk_size=200,chunk_overlap=50)


# 3. Create embedding for every chunk
chunk_embeddings = []

for chunk in chunks:
    embedding = generate_embedding(chunk)
    chunk_embeddings.append(embedding)


# 4. User question
question = "What major risks does the company face?"


# 5. Convert question into embedding
question_embedding = generate_embedding(question)


# 6. Compare question with every chunk
results = []

for index, chunk_embedding in enumerate(chunk_embeddings):

    similarity = cosine_similarity(question_embedding,chunk_embedding)

    results.append((similarity, index))


# 7. Sort from most similar to least similar
results.sort(reverse=True)


# 8. Display results
for similarity, index in results:

    print(f"\nSimilarity: {similarity:.4f}")
    print(f"Chunk {index + 1}:")
    print(chunks[index])