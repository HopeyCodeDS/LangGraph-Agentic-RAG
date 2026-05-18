from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from .nodes import create_agent_node, create_rewrite_node, create_generate_node
from .edges import create_grade_documents, route_after_agent

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]

def build_graph(tools):
    workflow = StateGraph(GraphState)

    agent_node = create_agent_node(tools)
    rewrite_node = create_rewrite_node()
    generate_node = create_generate_node()
    grade_documents_func = create_grade_documents()

    retrieve_node = ToolNode(tools)

    workflow.add_node("agent", agent_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "retrieve": "retrieve",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "agent",
        grade_documents_func,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )

    workflow.add_edge("rewrite", "agent")
    workflow.add_edge("generate", END)

    app = workflow.compile()

    return app
