
import os

from dotenv import load_dotenv
from ollama import chat

from src.utils.logger import get_logger


load_dotenv()

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:4b-instruct"
)

logger = get_logger(__name__)


def generate_response(
    prompt: str,
    system_prompt: str = ""
) -> str:

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
        messages=messages,
        options={
            "temperature": 0,
            "num_predict": 512
        }
    )

    # Token counts
    prompt_tokens = response.get(
        "prompt_eval_count", 0
    ) or 0

    generated_tokens = response.get(
        "eval_count", 0
    ) or 0

    # Timing metrics (nanoseconds)
    load_duration = response.get(
        "load_duration", 0
    ) or 0

    prompt_duration = response.get(
        "prompt_eval_duration", 0
    ) or 0

    generation_duration = response.get(
        "eval_duration", 0
    ) or 0

    total_duration = response.get(
        "total_duration", 0
    ) or 0

    # Convert nanoseconds to seconds
    load_seconds = load_duration / 1_000_000_000
    prompt_seconds = prompt_duration / 1_000_000_000
    generation_seconds = generation_duration / 1_000_000_000
    total_seconds = total_duration / 1_000_000_000

    tokens_per_second = (
        generated_tokens / generation_seconds
        if generation_seconds > 0
        else 0
    )

    logger.info("Ollama model: %s", OLLAMA_MODEL)

    logger.info("Prompt tokens: %d", prompt_tokens)
    logger.info("Generated tokens: %d", generated_tokens)

    logger.info(
        "Model loading time: %.3f seconds",
        load_seconds
    )

    logger.info(
        "Prompt processing time: %.3f seconds",
        prompt_seconds
    )

    logger.info(
        "Output generation time: %.3f seconds",
        generation_seconds
    )

    logger.info(
        "Total Ollama time: %.3f seconds",
        total_seconds
    )

    logger.info(
        "Generation speed: %.2f tokens/sec",
        tokens_per_second
    )

    return response["message"]["content"]
