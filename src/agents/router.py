from src.llm.provider import generate_response


def classify_route(question: str) -> str:

    system_prompt = """
You are a routing component in an enterprise AI system.

Your job is to decide which capabilities are required
to answer the user's question.

Available routes:

rag:
Use when the question asks for information, explanation,
facts, methodology, definitions, or evidence contained
in enterprise documents.

sql:
Use when the question requires calculating, aggregating,
filtering, counting, averaging, grouping, or analyzing
structured business data.

both:
Use when the question contains multiple requirements and
requires BOTH document knowledge and structured database
analysis.

Examples:

Question:
Why was a star schema chosen?
Route:
rag

Question:
How many orders are there?
Route:
sql

Question:
What does the report say about repeat customers, and how
many orders are in the database?
Route:
both

Rules:
- Return exactly one word.
- Valid responses are only: rag, sql, both.
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

    if route not in {
        "rag",
        "sql",
        "both"
    }:
        raise ValueError(
            f"Invalid route returned by LLM: {route}"
        )

    return route