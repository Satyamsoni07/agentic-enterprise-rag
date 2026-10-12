from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from src.utils.logger import get_logger

from src.rag.pipeline import answer_question
from src.agents.router import classify_route
from src.agents.planner import decompose_question
from src.tools.sql_agent_tool import run_sql_tool
from src.llm.provider import generate_response
from src.agents.verifier import verify_answer
from src.agents.corrector import correct_answer
from src.utils.timer import start_timer, elapsed_time

logger = get_logger(__name__)

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

    logger.info("Router classification started")

    # Start measuring router execution time
    start = start_timer()

    try:
        route = classify_route(question)

    except Exception:
        duration = elapsed_time(start)

        logger.exception(
            "Router classification failed after %.3f seconds",
            duration
        )
        raise

    # Calculate router latency
    duration = elapsed_time(start)

    logger.info(
        "Router classification time: %.3f seconds",
        duration
    )

    logger.info(
        "Router selected route: %s",
        route
    )

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

    logger.info("RAG retrieval started")

    start = start_timer()

    result = answer_question(question)

    duration = elapsed_time(start)

    logger.info("RAG retrieval completed")

    logger.info(
        "RAG execution time: %.3f seconds",
        duration
    )

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

    logger.info("SQL execution started")

    start = start_timer()

    try:
        result = run_sql_tool(question)

    except Exception:
        duration = elapsed_time(start)

        logger.exception(
            "SQL tool raised an exception after %.3f seconds",
            duration
        )
        raise

    duration = elapsed_time(start)

    logger.info(
        "SQL execution time: %.3f seconds",
        duration
    )

    if not result["success"]:

        logger.warning(
            "SQL execution failed: %s",
            result["error"]
        )

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

    logger.info("SQL execution completed successfully")

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

    logger.info("Question decomposition started")

    # Start measuring planner latency
    start = start_timer()

    try:
        plan = decompose_question(question)

    except Exception:
        duration = elapsed_time(start)

        logger.exception(
            "Question decomposition failed after %.3f seconds",
            duration
        )
        raise

    # Calculate planner execution time
    duration = elapsed_time(start)

    logger.info(
        "Question decomposition completed in %.3f seconds",
        duration
    )

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

    logger.info("Combined RAG retrieval started")

    start = start_timer()

    try:
        result = answer_question(question)

    except Exception:
        duration = elapsed_time(start)

        logger.exception(
            "Combined RAG failed after %.3f seconds",
            duration
        )
        raise

    duration = elapsed_time(start)

    logger.info("Combined RAG retrieval completed")

    logger.info(
        "Combined RAG execution time: %.3f seconds",
        duration
    )

    logger.info(
        "Number of retrieved sources: %d",
        len(result["sources"])
    )

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

    logger.info("Combined SQL execution started")

    start = start_timer()

    try:
        result = run_sql_tool(question)

    except Exception:
        duration = elapsed_time(start)

        logger.exception(
            "Combined SQL failed after %.3f seconds",
            duration
        )
        raise

    duration = elapsed_time(start)

    logger.info(
        "Combined SQL execution time: %.3f seconds",
        duration
    )

    if not result["success"]:

        logger.warning(
            "Combined SQL execution failed: %s",
            result["error"]
        )

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

    logger.info(
        "Combined SQL execution completed successfully"
    )

    logger.info(
        "Number of SQL result rows: %d",
        len(result["result"])
    )

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

    logger.info("Answer synthesis started")

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

    # Start measuring synthesis latency
    start = start_timer()

    try:
        answer = generate_response(
            user_prompt,
            system_prompt
        )

    except Exception:
        duration = elapsed_time(start)

        logger.exception(
            "Answer synthesis failed after %.3f seconds",
            duration
        )
        raise

    # Calculate execution time
    duration = elapsed_time(start)

    logger.info(
        "Answer synthesis completed in %.3f seconds",
        duration
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

    logger.info("Answer verification started")

    evidence = build_combined_evidence(state)

    start = start_timer()

    try:
        verification = verify_answer(
            question=state["question"],
            evidence=evidence,
            answer=state["answer"]
        )

    except Exception:
        duration = elapsed_time(start)

        logger.exception(
            "Answer verification raised an error after %.3f seconds",
            duration
        )
        raise

    duration = elapsed_time(start)

    logger.info(
        "Answer verification time: %.3f seconds",
        duration
    )

    supported = verification["supported"]
    reason = verification["reason"]

    if supported:
        logger.info("Answer verification passed")

    else:
        logger.warning("Answer verification failed")

        logger.warning(
            "Verification reason: %s",
            reason
        )

    return {
        "verification_supported": supported,
        "verification_reason": reason,
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

# 12. Corrector node
def corrector_node(state: RouterState) -> dict:

    attempt_number = state["correction_attempts"] + 1

    logger.info(
        "Answer correction started | Attempt: %d",
        attempt_number
    )

    evidence = build_combined_evidence(state)

    start = start_timer()

    try:
        corrected_answer = correct_answer(
            question=state["question"],
            evidence=evidence,
            answer=state["answer"],
            verification_reason=state[
                "verification_reason"
            ]
        )

    except Exception:
        duration = elapsed_time(start)

        logger.exception(
            "Answer correction failed after %.3f seconds | Attempt: %d",
            duration,
            attempt_number
        )
        raise

    duration = elapsed_time(start)

    logger.info(
        "Answer correction completed in %.3f seconds | Attempt: %d",
        duration,
        attempt_number
    )

    return {
        "answer": corrected_answer,
        "correction_attempts": attempt_number,
        "execution_trace": (
            state["execution_trace"]
            + ["corrector"]
        )
    }



# 13. Safe-stop node
def safe_stop_node(state: RouterState) -> dict:

    logger.warning(
        "Agent stopped: Answer remains unsupported "
        "after %d correction attempt(s)",
        state["correction_attempts"]
    )

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