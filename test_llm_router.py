from src.agents.router import classify_route


questions = [
    "Why was a star schema chosen for the project?",

    "How many orders are there?",

    (
        "What does the report say about repeat customers, "
        "and how many orders are in the database?"
    )
]


for question in questions:

    route = classify_route(question)

    print("\n" + "=" * 60)

    print("Question:")
    print(question)

    print("\nRoute:")
    print(route)