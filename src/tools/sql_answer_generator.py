from src.llm.provider import generate_response


def generate_sql_answer(
    question: str,
    sql: str,
    result: list
) -> str:

    system_prompt = """
You are a data analysis assistant.

Answer the user's question using only the SQL query
and database result provided to you.

Rules:
- Use only the provided database result.
- Do not invent values or facts.
- Do not use outside knowledge.
- Do not assume units or currencies.
- If currency is not explicitly provided, do not add
  symbols such as $, ₹, €, or £.
- Preserve the meaning of the returned values.
- Present the result clearly and concisely.
- If multiple rows are returned, format them so they
  are easy to read.
- Do not mention implementation details unless necessary.
"""

    user_prompt = f"""
User question:
{question}

Executed SQL:
{sql}

Database result:
{result}

Provide a clear answer to the user's question.
"""

    answer = generate_response(
        user_prompt,
        system_prompt
    )

    return answer.strip()