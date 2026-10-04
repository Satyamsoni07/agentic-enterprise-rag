from src.tools.sql_agent_tool import run_sql_tool


questions = [
    "Show total sales by product.",
    "Which customer spent the most?",
    "How many orders are there?"
]


for question in questions:

    result = run_sql_tool(question)

    print("\n" + "=" * 60)

    print("Question:")
    print(question)

    print("\nSuccess:")
    print(result["success"])

    print("\nGenerated SQL:")
    print(result["sql"])

    print("\nResult:")
    print(result["result"])

    print("\nError:")
    print(result["error"])