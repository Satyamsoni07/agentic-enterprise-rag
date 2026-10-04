from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.agents.verifier import verify_answer
from src.agents.corrector import correct_answer


class TestState(TypedDict):
    question: str
    evidence: str
    answer: str

    verification_supported: bool
    verification_reason: str

    correction_attempts: int


def bad_answer_node(state: TestState) -> dict:

    return {
        "answer": (
            "There are 10 orders in the database "
            "with total revenue of $500,000."
        )
    }


def verifier_node(state: TestState) -> dict:

    verification = verify_answer(
        question=state["question"],
        evidence=state["evidence"],
        answer=state["answer"]
    )

    return {
        "verification_supported": verification["supported"],
        "verification_reason": verification["reason"]
    }


def choose_verification_route(
    state: TestState
) -> str:

    if state["verification_supported"]:
        return "pass"

    if state["correction_attempts"] >= 1:
        return "stop"

    return "correct"


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
        )
    }


def safe_stop_node(state: TestState) -> dict:

    return {
        "answer": (
            "I could not produce a sufficiently verified "
            "answer from the available evidence."
        )
    }


graph_builder = StateGraph(TestState)


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


graph_builder.add_edge(
    START,
    "bad_answer"
)

graph_builder.add_edge(
    "bad_answer",
    "verifier"
)


graph_builder.add_conditional_edges(
    "verifier",
    choose_verification_route,
    {
        "pass": END,
        "correct": "corrector",
        "stop": "safe_stop"
    }
)


graph_builder.add_edge(
    "corrector",
    "verifier"
)

graph_builder.add_edge(
    "safe_stop",
    END
)


test_graph = graph_builder.compile()


initial_state = {
    "question": "How many orders are in the database?",

    "evidence": """
Executed SQL:
SELECT COUNT(*) FROM orders;

Raw SQL result:
[(6,)]
""",

    "answer": "",

    "verification_supported": False,
    "verification_reason": "",

    "correction_attempts": 0
}


result = test_graph.invoke(
    initial_state
)


print("\nFINAL ANSWER:")
print(result["answer"])

print("\nVERIFICATION SUPPORTED:")
print(result["verification_supported"])

print("\nVERIFICATION REASON:")
print(result["verification_reason"])

print("\nCORRECTION ATTEMPTS:")
print(result["correction_attempts"])