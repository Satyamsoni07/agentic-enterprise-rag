from src.agents.simple_graph import simple_graph


initial_state = {
    "question": "What is RAG?",
    "message": "",
    "final_message": ""
}


result = simple_graph.invoke(
    initial_state
)


print("\nFinal State:")
print(result)

print("\nFinal Message:")
print(result["final_message"])