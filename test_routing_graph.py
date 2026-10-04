from src.agents.routing_graph import routing_graph


questions = [
    # RAG only
    "Why was a star schema chosen for the project?",

    # SQL only
    "Show total sales by product.",

    # BOTH
    (
        "What does the report say about repeat customers, "
        "and how many orders are in the database?"
    )
]


for question in questions:

    print("\n" + "=" * 70)
    print("QUESTION:")
    print(question)

    initial_state = {
        "question": question,
        "route": "",

        "rag_question": "",
        "sql_question": "",

        "rag_answer": "",
        "sql_answer": "",

        "answer": "",
        "sources": [],

        "sql": "",
        "sql_result": [],

        "verification_supported": False,
        "verification_reason": "",

        "correction_attempts": 0

    }

    result = routing_graph.invoke(
        initial_state
    )

    print("\nROUTE:")
    print(result["route"])

    print("\nFINAL ANSWER:")
    print(result["answer"])

    # Show document evidence
    if result["sources"]:

        print("\nSOURCES:")

        for source in result["sources"]:
            print(source)

    # Show SQL evidence
    if result["sql"]:

        print("\nGENERATED SQL:")
        print(result["sql"])

        print("\nRAW SQL RESULT:")
        print(result["sql_result"])

    # Show decomposition for BOTH route
    if result["route"] == "both":

        print("\nRAG SUB-QUESTION:")
        print(result["rag_question"])

        print("\nSQL SUB-QUESTION:")
        print(result["sql_question"])

        print("\nRAG ANSWER:")
        print(result["rag_answer"])

        print("\nSQL ANSWER:")
        print(result["sql_answer"])

        print("\nVERIFICATION SUPPORTED:")
        print(result["verification_supported"])

        print("\nVERIFICATION REASON:")
        print(result["verification_reason"])