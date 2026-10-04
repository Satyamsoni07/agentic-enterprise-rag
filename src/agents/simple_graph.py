from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# 1. Define the state
class AgentState(TypedDict):
    question: str
    message: str
    final_message: str


# 2. First node
def greeting_node(state: AgentState) -> dict:

    question = state["question"]

    return {"message": f"I received your question: {question}"}


# 3. Second node
def final_node(state: AgentState) -> dict:

    previous_message = state["message"]

    return {
        "final_message": (
            f"{previous_message} "
            "The workflow is now complete."
        )
    }


# 4. Create graph
graph_builder = StateGraph(AgentState)


# 5. Add nodes
graph_builder.add_node(
    "greeting",
    greeting_node
)

graph_builder.add_node(
    "final",
    final_node
)


# 6. Connect nodes
graph_builder.add_edge(
    START,
    "greeting"
)

graph_builder.add_edge(
    "greeting",
    "final"
)

graph_builder.add_edge(
    "final",
    END
)


# 7. Compile graph
simple_graph = graph_builder.compile()