from src.llm.provider import generate_response


def correct_answer(
    question: str,
    evidence: str,
    answer: str,
    verification_reason: str
) -> str:

    system_prompt = """
You are a correction component in an enterprise AI system.

The previous answer failed verification.

Your job is to produce a corrected answer using only the
provided evidence.

Rules:
- Use only the provided evidence.
- Do not use outside knowledge.
- Remove or correct unsupported factual claims.
- Do not invent values, facts, units, or currencies.
- Preserve valid document citations when they correctly
  support a claim.
- Do not attach document citations to SQL-derived claims.
- Answer the user's original question clearly and concisely.
"""

    user_prompt = f"""
User question:
{question}

Evidence:
{evidence}

Previous answer:
{answer}

Verification failure reason:
{verification_reason}

Produce a corrected answer.
"""

    response = generate_response(
        user_prompt,
        system_prompt
    )

    return response.strip()