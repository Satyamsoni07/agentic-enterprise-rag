from ollama import chat

response = chat(
    model="qwen3:4b-instruct",
    messages=[
        {
            "role": "user",
            "content": "In Generative AI, explain RAG in one simple sentence."
        }
    ]
)

print(response["message"]["content"])