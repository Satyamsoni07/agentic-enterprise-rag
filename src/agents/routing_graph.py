from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.rag.pipeline import answer_question
from src.agents.router import classify_route
from src.agents.planner import decompose_question
from src.tools.sql_agent_tool import run_sql_tool
from src.llm.provider import generate_response
from src.agents.verifier import verify_answer
from src.agents.corrector import correct_answer


# 1. Define state
class RouterState(TypedDict):
    question: str
    route: str

    rag_question: str
    sql_question: str

    rag_answer: str
    sql_answer: str

    answer: str
    sources: list

    sql: str
    sql_result: list

    verification_supported: bool
    verification_reason: str

    correction_attempts: int

    execution_trace: list


# 2. Router node
def router_node(state: RouterState) -> dict:

    question = state["question"]

    route = classify_route(question)

    return {
        "route": route,
        "execution_trace": (
            state["execution_trace"]
            + ["router"]
        )
    }


# 3. Normal RAG node
def rag_node(state: RouterState) -> dict:

    question = state["question"]

    result = answer_question(question)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "sql": "",
        "sql_result": [],
        "execution_trace": (
            state["execution_trace"]
            + ["rag"]
        )
    }


# 4. Normal SQL node
def sql_node(state: RouterState) -> dict:

    question = state["question"]

    result = run_sql_tool(question)

    if not result["success"]:
        return {
            "answer": (
                f"SQL tool failed: {result['error']}"
            ),
            "sources": [],
            "sql": result["sql"],
            "sql_result": [],
            "execution_trace": (
                state["execution_trace"]
                + ["sql"]
            )
        }

    return {
        "answer": result["answer"],
        "sources": [],
        "sql": result["sql"],
        "sql_result": result["result"],
        "execution_trace": (
            state["execution_trace"]
            + ["sql"]
        )
    }


# 5. Planner node for BOTH route
def planner_node(state: RouterState) -> dict:

    question = state["question"]

    plan = decompose_question(question)

    return {
        "rag_question": plan["rag_question"],
        "sql_question": plan["sql_question"],
        "execution_trace": (
            state["execution_trace"]
            + ["planner"]
        )
    }


# 6. RAG node for BOTH route
def combined_rag_node(state: RouterState) -> dict:

    question = state["rag_question"]

    result = answer_question(question)

    return {
        "rag_answer": result["answer"],
        "sources": result["sources"],
        "execution_trace": (
            state["execution_trace"]
            + ["combined_rag"]
        )
    }


# 7. SQL node for BOTH route
def combined_sql_node(state: RouterState) -> dict:

    question = state["sql_question"]

    result = run_sql_tool(question)

    if not result["success"]:
        return {
            "sql_answer": (
                f"SQL tool failed: {result['error']}"
            ),
            "sql": result["sql"],
            "sql_result": [],
            "execution_trace": (
                state["execution_trace"]
                + ["combined_sql"]
            )
        }

    return {
        "sql_answer": result["answer"],
        "sql": result["sql"],
        "sql_result": result["result"],
        "execution_trace": (
            state["execution_trace"]
            + ["combined_sql"]
        )
    }


# 8. Final synthesis node
def synthesis_node(state: RouterState) -> dict:

    original_question = state["question"]
    rag_answer = state["rag_answer"]
    sql_answer = state["sql_answer"]

    system_prompt = """
You are the final synthesis component of an enterprise AI system.

Answer the user's original question using only the RAG answer
and SQL answer provided.

Rules:
- Combine both answers into one clear response.
- Do not invent additional facts.
- Preserve citations such as [1], [2] only for claims that
  came from the RAG answer.
- Never attach RAG citations to information obtained from
  the SQL answer.
- SQL-derived claims must not receive document citations.
- Do not invent new citations.
- Do not assume currencies, units, or other metadata that
  are not explicitly provided.
- Make sure both parts of the user's question are answered.
"""

    user_prompt = f"""
Original question:
{original_question}

RAG answer:
{rag_answer}

SQL answer:
{sql_answer}

Create the final combined answer.
"""

    answer = generate_response(
        user_prompt,
        system_prompt
    )

    return {
        "answer": answer.strip(),
        "execution_trace": (
            state["execution_trace"]
            + ["synthesis"]
        )
    }


# 9. Build evidence for verification
def build_combined_evidence(
    state: RouterState
) -> str:

    rag_answer = state["rag_answer"]
    sql = state["sql"]
    sql_result = state["sql_result"]

    evidence = f"""
Document-grounded RAG result:
{rag_answer}

Executed SQL:
{sql}

Raw SQL result:
{sql_result}
"""

    return evidence


# 10. Verifier node
def verifier_node(state: RouterState) -> dict:

    evidence = build_combined_evidence(state)

    verification = verify_answer(
        question=state["question"],
        evidence=evidence,
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


# 11. Verification routing function
def choose_verification_route(
    state: RouterState
) -> str:

    # Answer is supported
    if state["verification_supported"]:
        return "pass"

    # One correction has already been attempted
    if state["correction_attempts"] >= 1:
        return "stop"

    # First verification failure
    return "correct"


# 12. Corrector node
def corrector_node(state: RouterState) -> dict:

    evidence = build_combined_evidence(state)

    corrected_answer = correct_answer(
        question=state["question"],
        evidence=evidence,
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


# 13. Safe-stop node
def safe_stop_node(state: RouterState) -> dict:

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


# 14. Main routing function
def choose_route(state: RouterState) -> str:

    return state["route"]


# 15. Create graph
graph_builder = StateGraph(
    RouterState
)


# 16. Add nodes
graph_builder.add_node(
    "router",
    router_node
)

graph_builder.add_node(
    "rag",
    rag_node
)

graph_builder.add_node(
    "sql",
    sql_node
)

graph_builder.add_node(
    "planner",
    planner_node
)

graph_builder.add_node(
    "combined_rag",
    combined_rag_node
)

graph_builder.add_node(
    "combined_sql",
    combined_sql_node
)

graph_builder.add_node(
    "synthesis",
    synthesis_node
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


# 17. START → Router
graph_builder.add_edge(
    START,
    "router"
)


# 18. Main conditional routing
graph_builder.add_conditional_edges(
    "router",
    choose_route,
    {
        "rag": "rag",
        "sql": "sql",
        "both": "planner"
    }
)


# 19. BOTH route workflow
graph_builder.add_edge(
    "planner",
    "combined_rag"
)

graph_builder.add_edge(
    "combined_rag",
    "combined_sql"
)

graph_builder.add_edge(
    "combined_sql",
    "synthesis"
)


# 20. Synthesis → Verifier
graph_builder.add_edge(
    "synthesis",
    "verifier"
)


# 21. Verification decision
graph_builder.add_conditional_edges(
    "verifier",
    choose_verification_route,
    {
        "pass": END,
        "correct": "corrector",
        "stop": "safe_stop"
    }
)


# 22. Corrector → Verifier again
graph_builder.add_edge(
    "corrector",
    "verifier"
)


# 23. Safe stop → END
graph_builder.add_edge(
    "safe_stop",
    END
)


# 24. Normal single-tool paths
graph_builder.add_edge(
    "rag",
    END
)

graph_builder.add_edge(
    "sql",
    END
)


# 25. Compile graph
routing_graph = graph_builder.compile()