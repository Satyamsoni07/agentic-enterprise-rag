from src.agents.corrector import correct_answer


question = "How many orders are in the database?"

evidence = """
SQL query:
SELECT COUNT(*) FROM orders;

SQL result:
[(6,)]
"""

bad_answer = """
There are 10 orders in the database,
with total revenue of $500,000.
"""

verification_reason = """
The evidence shows 6 orders, not 10.
The evidence also does not provide total revenue.
"""


corrected_answer = correct_answer(
    question=question,
    evidence=evidence,
    answer=bad_answer,
    verification_reason=verification_reason
)


print("\nBAD ANSWER:")
print(bad_answer)

print("\nCORRECTED ANSWER:")
print(corrected_answer)