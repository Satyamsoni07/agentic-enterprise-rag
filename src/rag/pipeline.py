
from src.rag.retriever import retrieve_documents
from src.rag.reranker import rerank_documents
from src.llm.provider import generate_response

from src.utils.logger import get_logger
from src.utils.timer import start_timer, elapsed_time


logger = get_logger(__name__)


def answer_question(
    question: str,
    retrieval_k: int = 10,
    final_k: int = 3,
    max_distance: float | None = None
) -> dict:

    # 1. Retrieve candidate documents using vector search
    logger.info("Vector retrieval started")

    start = start_timer()

    try:
        retrieved_documents = retrieve_documents(
            question=question,
            top_k=retrieval_k,
            max_distance=max_distance
        )

    except Exception:
        logger.exception(
            "Vector retrieval failed after %.3f seconds",
            elapsed_time(start)
        )
        raise

    logger.info(
        "Vector retrieval completed in %.3f seconds",
        elapsed_time(start)
    )

    logger.info(
        "Number of retrieved candidates: %d",
        len(retrieved_documents)
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
    logger.info("Cross-encoder reranking started")

    start = start_timer()

    try:
        retrieved_documents = rerank_documents(
            question=question,
            documents=retrieved_documents,
            top_k=final_k,
            min_score=0.0
        )

    except Exception:
        logger.exception(
            "Reranking failed after %.3f seconds",
            elapsed_time(start)
        )
        raise

    logger.info(
        "Cross-encoder reranking completed in %.3f seconds",
        elapsed_time(start)
    )

    logger.info(
        "Number of selected chunks: %d",
        len(retrieved_documents)
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
    logger.info("RAG LLM generation started")

    start = start_timer()

    try:
        answer = generate_response(
            user_prompt,
            system_prompt
        )

    except Exception:
        logger.exception(
            "RAG LLM generation failed after %.3f seconds",
            elapsed_time(start)
        )
        raise

    logger.info(
        "RAG LLM generation completed in %.3f seconds",
        elapsed_time(start)
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
