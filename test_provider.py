from src.llm.provider import generate_response


system_prompt = """
You are a beginner-friendly AI teacher.

Rules:
- Explain concepts in simple language.
- Keep the answer short.
- Give one everyday analogy.
"""


question = "What is an embedding in Generative AI?"


answer = generate_response(
    question,
    system_prompt
)


print(answer)