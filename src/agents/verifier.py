import json

from src.llm.provider import generate_response


def verify_answer(
    question: str,
    evidence: str,
    answer: str
) -> dict:

    system_prompt = """
You are a verification component in an enterprise AI system.

Your job is to determine whether the final answer is supported
by the provided evidence.

Check whether factual claims in the answer can be supported
by the evidence.

Return valid JSON using exactly this structure:

{
    "supported": true,
    "reason": "short explanation"
}

Rules:
- Use only the provided evidence.
- Do not use outside knowledge.
- If the answer contains an important factual claim that is
  not supported by the evidence, set supported to false.
- If the answer contradicts the evidence, set supported to false.
- Do not fail an answer merely because wording differs from
  the evidence.
- Do not answer the original question yourself.
- Return only valid JSON.
- Do not use Markdown code fences.
"""

    user_prompt = f"""
User question:
{question}

Evidence:
{evidence}

Final answer:
{answer}

Verify the final answer.
"""

    response = generate_response(
        user_prompt,
        system_prompt
    )

    verification = json.loads(
        response.strip()
    )

    return verification