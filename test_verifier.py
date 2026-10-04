from src.agents.verifier import verify_answer


question = "How many orders are in the database?"

evidence = """
SQL query:
SELECT COUNT(*) FROM orders;

SQL result:
[(6,)]
"""


good_answer = """
There are 6 orders in the database.
"""


bad_answer = """
There are 10 orders in the database,
with total revenue of $500,000.
"""


print("\nGOOD ANSWER TEST")

result = verify_answer(
    question=question,
    evidence=evidence,
    answer=good_answer
)

print(result)


print("\nBAD ANSWER TEST")

result = verify_answer(
    question=question,
    evidence=evidence,
    answer=bad_answer
)

print(result)