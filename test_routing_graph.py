from src.agents.routing_graph import routing_graph


questions = [
    "Why was a star schema chosen for the project?",
    "Show total sales by product."
]


for question in questions:

    initial_state = {
        "question": question,
        "route": "",
        "answer": "",
        "sources": [],
        "sql": "",
        "sql_result": []
    }

    result = routing_graph.invoke(
        initial_state
    )

    print("\n" + "=" * 60)

    print("\nQuestion:")
    print(result["question"])

    print("\nSelected Route:")
    print(result["route"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    print(result["sources"])

    # Show SQL evidence only when SQL route was selected
    if result["route"] == "sql":

        print("\nGenerated SQL:")
        print(result["sql"])

        print("\nRaw SQL Result:")
        print(result["sql_result"])