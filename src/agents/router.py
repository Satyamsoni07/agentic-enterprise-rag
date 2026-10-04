from src.llm.provider import generate_response


def classify_route(question: str) -> str:

    system_prompt = """
You are a routing component in an enterprise AI system.

Your job is to decide which tool should handle the user's question.

Available routes:

rag:
Use when the question asks for information, explanation, facts,
definitions, methodology, or evidence contained in enterprise documents.

sql:
Use when the question requires calculating, aggregating, filtering,
counting, averaging, grouping, or analyzing structured business data.

Rules:
- Return exactly one word.
- Valid responses are only: rag or sql
- Do not provide explanations.
"""

    user_prompt = f"""
Question:
{question}

Choose the correct route.
"""

    response = generate_response(
        user_prompt,
        system_prompt
    )

    route = response.strip().lower()

    if route not in {"rag", "sql"}:
        raise ValueError(
            f"Invalid route returned by LLM: {route}"
        )

    return route