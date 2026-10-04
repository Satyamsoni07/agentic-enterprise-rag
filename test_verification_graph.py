from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.agents.verifier import verify_answer
from src.agents.corrector import correct_answer


# 1. Define test state
class TestState(TypedDict):
    question: str
    evidence: str
    answer: str

    verification_supported: bool
    verification_reason: str

    correction_attempts: int

    execution_trace: list


# 2. Create intentionally wrong answer
def bad_answer_node(state: TestState) -> dict:

    return {
        "answer": (
            "There are 10 orders in the database "
            "with total revenue of $500,000."
        ),
        "execution_trace": (
            state["execution_trace"]
            + ["bad_answer"]
        )
    }


# 3. Verify the answer
def verifier_node(state: TestState) -> dict:

    verification = verify_answer(
        question=state["question"],
        evidence=state["evidence"],
        answer=state["answer"]
    )

    return {
        "verification_supported": verification[
            "supported"
        ],
        "verification_reason": verification[
            "reason"
        ],
        "execution_trace": (
            state["execution_trace"]
            + ["verifier"]
        )
    }


# 4. Decide what to do after verification
def choose_verification_route(
    state: TestState
) -> str:

    # Answer passed verification
    if state["verification_supported"]:
        return "pass"

    # Correction was already attempted once
    if state["correction_attempts"] >= 1:
        return "stop"

    # First verification failure
    return "correct"


# 5. Correct unsupported answer
def corrector_node(state: TestState) -> dict:

    corrected_answer = correct_answer(
        question=state["question"],
        evidence=state["evidence"],
        answer=state["answer"],
        verification_reason=state[
            "verification_reason"
        ]
    )

    return {
        "answer": corrected_answer,
        "correction_attempts": (
            state["correction_attempts"] + 1
        ),
        "execution_trace": (
            state["execution_trace"]
            + ["corrector"]
        )
    }


# 6. Safe stop if correction also fails
def safe_stop_node(state: TestState) -> dict:

    return {
        "answer": (
            "I could not produce a sufficiently verified "
            "answer from the available evidence."
        ),
        "execution_trace": (
            state["execution_trace"]
            + ["safe_stop"]
        )
    }


# 7. Create graph
graph_builder = StateGraph(
    TestState
)


# 8. Add nodes
graph_builder.add_node(
    "bad_answer",
    bad_answer_node
)

graph_builder.add_node(
    "verifier",
    verifier_node
)

graph_builder.add_node(
    "corrector",
    corrector_node
)

graph_builder.add_node(
    "safe_stop",
    safe_stop_node
)


# 9. START → Bad answer
graph_builder.add_edge(
    START,
    "bad_answer"
)


# 10. Bad answer → Verifier
graph_builder.add_edge(
    "bad_answer",
    "verifier"
)


# 11. Verification decision
graph_builder.add_conditional_edges(
    "verifier",
    choose_verification_route,
    {
        "pass": END,
        "correct": "corrector",
        "stop": "safe_stop"
    }
)


# 12. Corrector → Verifier again
graph_builder.add_edge(
    "corrector",
    "verifier"
)


# 13. Safe stop → END
graph_builder.add_edge(
    "safe_stop",
    END
)


# 14. Compile graph
test_graph = graph_builder.compile()


# 15. Initial state
initial_state = {
    "question": (
        "How many orders are in the database?"
    ),

    "evidence": """
Executed SQL:
SELECT COUNT(*) FROM orders;

Raw SQL result:
[(6,)]
""",

    "answer": "",

    "verification_supported": False,
    "verification_reason": "",

    "correction_attempts": 0,

    "execution_trace": []
}


# 16. Run graph
result = test_graph.invoke(
    initial_state
)


# 17. Print results
print("\nFINAL ANSWER:")
print(result["answer"])

print("\nVERIFICATION SUPPORTED:")
print(result["verification_supported"])

print("\nVERIFICATION REASON:")
print(result["verification_reason"])

print("\nCORRECTION ATTEMPTS:")
print(result["correction_attempts"])

print("\nEXECUTION TRACE:")
print(result["execution_trace"])