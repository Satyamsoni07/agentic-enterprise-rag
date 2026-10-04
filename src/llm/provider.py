import os

from dotenv import load_dotenv

from src.llm.ollama_provider import generate_response as ollama_generate
from src.llm.openai_provider import generate_response as openai_generate


load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")


def generate_response(prompt: str, system_prompt: str = "") -> str:

    if LLM_PROVIDER == "ollama":
        return ollama_generate(prompt, system_prompt)

    if LLM_PROVIDER == "openai":
        return openai_generate(prompt, system_prompt)

    raise ValueError(
        f"Unsupported LLM provider: {LLM_PROVIDER}"
    )