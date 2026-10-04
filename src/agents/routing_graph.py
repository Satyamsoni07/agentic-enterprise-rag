from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.rag.pipeline import answer_question
from src.agents.router import classify_route
from src.tools.sql_agent_tool import run_sql_tool


# 1. Define state
class RouterState(TypedDict):
    question: str
    route: str
    answer: str
    sources: list
    sql: str
    sql_result: list


# 2. Router node
def router_node(state: RouterState) -> dict:

    question = state["question"]

    route = classify_route(question)

    return {
        "route": route
    }


# 3. REAL RAG node
def rag_node(state: RouterState) -> dict:

    question = state["question"]

    result = answer_question(question)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "sql": "",
        "sql_result": []
    }


# 4. SQL node
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
            "sql_result": []
        }

    return {
        "answer": result["answer"],
        "sources": [],
        "sql": result["sql"],
        "sql_result": result["result"]
    }


# 5. Routing function
def choose_route(state: RouterState) -> str:

    return state["route"]


# 6. Create graph
graph_builder = StateGraph(
    RouterState
)


# 7. Add nodes
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


# 8. START → Router
graph_builder.add_edge(
    START,
    "router"
)


# 9. Conditional routing
graph_builder.add_conditional_edges(
    "router",
    choose_route,
    {
        "rag": "rag",
        "sql": "sql"
    }
)


# 10. Finish paths
graph_builder.add_edge(
    "rag",
    END
)

graph_builder.add_edge(
    "sql",
    END
)


# 11. Compile graph
routing_graph = graph_builder.compile()