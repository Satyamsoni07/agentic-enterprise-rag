import json

from src.llm.provider import generate_response


def decompose_question(question: str) -> dict:

    system_prompt = """
You are a planning component in an enterprise AI system.

The user has asked a question that requires both:
1. document retrieval using RAG
2. structured database analysis using SQL

Break the question into exactly two independent sub-questions.

Return valid JSON using exactly this structure:

{
    "rag_question": "...",
    "sql_question": "..."
}

Rules:
- rag_question must contain only the part that should be
  answered using enterprise documents.
- sql_question must contain only the part that should be
  answered using structured database data.
- Preserve the user's original meaning.
- Do not answer either question.
- Return only valid JSON.
- Do not use Markdown code fences.
"""

    user_prompt = f"""
User question:
{question}

Decompose the question.
"""

    response = generate_response(
        user_prompt,
        system_prompt
    )

    plan = json.loads(
        response.strip()
    )

    return plan