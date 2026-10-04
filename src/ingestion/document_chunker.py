from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(pages: list[dict],chunk_size: int = 500,chunk_overlap: int = 100) -> list[dict]:

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)

    chunked_documents = []

    for page in pages:

        chunks = splitter.split_text(page["text"])

        for chunk in chunks:

            chunked_documents.append(
                {
                    "text": chunk,
                    "metadata": page["metadata"].copy()
                }
            )

    return chunked_documents