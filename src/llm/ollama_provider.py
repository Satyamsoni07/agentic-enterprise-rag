import os

from dotenv import load_dotenv
from ollama import chat


load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")


def generate_response(prompt: str, system_prompt: str = "") -> str:  #"": menas system_prompt is optional.

    messages = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    response = chat(
        model=OLLAMA_MODEL,
        messages=messages
    )

    return response["message"]["content"]