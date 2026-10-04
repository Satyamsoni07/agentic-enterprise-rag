from src.rag.retriever import retrieve_documents
from src.rag.reranker import rerank_documents
from src.llm.provider import generate_response


def answer_question(
    question: str,
    retrieval_k: int = 10,
    final_k: int = 3,
    max_distance: float | None = None
) -> dict:

    # 1. Retrieve candidate documents using vector search
    retrieved_documents = retrieve_documents(
        question=question,
        top_k=retrieval_k,
        max_distance=max_distance
    )

    # If vector search finds no documents
    if not retrieved_documents:
        return {
            "answer": (
                "I could not find this information "
                "in the provided documents."
            ),
            "sources": []
        }

    # 2. Rerank the retrieved candidate documents
    retrieved_documents = rerank_documents(
        question=question,
        documents=retrieved_documents,
        top_k=final_k,
        min_score=0.0
    )

    # If reranker rejects all documents
    if not retrieved_documents:
        return {
            "answer": (
                "I could not find this information "
                "in the provided documents."
            ),
            "sources": []
        }

    # 3. Build numbered context for citation-aware generation
    context_parts = []

    for index, document in enumerate(
        retrieved_documents,
        start=1
    ):
        metadata = document["metadata"]

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get("page")

        context_part = f"""
[Source {index}]
Source: {source}
Page: {page}

{document["text"]}
"""

        context_parts.append(context_part)

    # 4. Combine all numbered evidence chunks
    context = "\n\n".join(context_parts)

    # 5. Grounding and citation instructions
    system_prompt = """
You are an enterprise document assistant.

Answer the user's question using only the provided context.

Rules:
- Do not use outside knowledge.
- Base factual claims only on the provided sources.
- Cite supporting evidence using [1], [2], etc.
- The citation number must correspond to the numbered source in the context.
- Do not cite a source unless it supports the claim.
- If multiple sources support the same claim, you may cite multiple sources such as [1][2].
- If the answer is not present in the context, say:
  "I could not find this information in the provided documents."
- Keep the answer clear and concise.
"""

    user_prompt = f"""
Context:

{context}

Question:
{question}
"""

    # 6. Generate the final grounded answer
    answer = generate_response(
        user_prompt,
        system_prompt
    )

    # 7. Collect sources in the same order
    # as the citation numbers
    sources = []

    for index, document in enumerate(
        retrieved_documents,
        start=1
    ):
        metadata = document["metadata"]

        source = {
            "citation_id": index,
            "source": metadata.get(
                "source",
                "Unknown"
            ),
            "page": metadata.get("page"),
            "rerank_score": document.get(
                "rerank_score"
            )
        }

        sources.append(source)

    # 8. Return answer and supporting sources
    return {
        "answer": answer,
        "sources": sources
    }