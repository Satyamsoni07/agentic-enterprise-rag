from src.agents.router import classify_route


questions = [
    "Why was a star schema chosen for the project?",
    "What is customer_unique_id?",
    "Calculate the average order value.",
    "Show total sales by month.",
    "How did the project prevent double counting?"
]


for question in questions:

    route = classify_route(question)

    print("\nQuestion:")
    print(question)

    print("Route:")
    print(route)