from src.tools.sql_generator import generate_sql


questions = [
    "Show total sales by product.",
    "What is the total sales amount?",
    "Which customer spent the most?",
    "How many orders are there?"
]


for question in questions:

    sql = generate_sql(question)

    print("\n" + "=" * 60)

    print("Question:")
    print(question)

    print("\nGenerated SQL:")
    print(sql)