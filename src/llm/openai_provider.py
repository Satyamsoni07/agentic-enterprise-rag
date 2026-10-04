import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6")

client = OpenAI()


def generate_response(prompt: str, system_prompt: str = "") -> str:

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=system_prompt if system_prompt else None,
        input=prompt
    )

    return response.output_text