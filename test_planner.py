from src.agents.planner import decompose_question


question = (
    "What does the report say about repeat customers, "
    "and how many orders are in the database?"
)


plan = decompose_question(
    question
)


print("\nOriginal Question:")
print(question)

print("\nRAG Sub-question:")
print(plan["rag_question"])

print("\nSQL Sub-question:")
print(plan["sql_question"])