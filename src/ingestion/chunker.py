def chunk_text(text: str,chunk_size: int = 200,chunk_overlap: int = 50) -> list[str]:

    chunks = []

    step_size = chunk_size - chunk_overlap

    for i in range(0, len(text), step_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks