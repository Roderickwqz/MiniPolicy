from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from app.core.state import RunState
from app.graph.nodes import (
    node_input,
    node_ingest_docs,
    node_vector_index,
    node_semantic_retrieve,
    node_graph_build,
    node_parallel_skills,
    node_debate,
    node_verify,
    node_score,
    node_report,
    node_evals,
)


def build_workflow():
    workflow = StateGraph(RunState)

    workflow.add_node("Input", node_input)
    workflow.add_node("IngestDocs", node_ingest_docs)
    workflow.add_node("VectorIndex", node_vector_index)
    workflow.add_node("SemanticRetrieve", node_semantic_retrieve)
    workflow.add_node("GraphBuild", node_graph_build)
    workflow.add_node("ParallelSkills", node_parallel_skills)
    workflow.add_node("Debate", node_debate)
    workflow.add_node("Verify", node_verify)
    workflow.add_node("Score", node_score)
    workflow.add_node("Report", node_report)
    workflow.add_node("Evals", node_evals)

    workflow.add_edge(START, "Input")
    workflow.add_edge("Input", "IngestDocs")
    workflow.add_edge("IngestDocs", "VectorIndex")
    workflow.add_edge("VectorIndex", "SemanticRetrieve")
    workflow.add_edge("SemanticRetrieve", "GraphBuild")
    workflow.add_edge("GraphBuild", "ParallelSkills")
    workflow.add_edge("ParallelSkills", "Debate")
    workflow.add_edge("Debate", "Verify")
    workflow.add_edge("Verify", "Score")
    workflow.add_edge("Score", "Report")
    workflow.add_edge("Report", "Evals")
    workflow.add_edge("Evals", END)

    # 这个函数只负责“定义图”，不负责“资源生命周期”
    # checkpointer = SqliteSaver.from_conn_string(sqlite_path)
    # graph = workflow.compile(checkpointer=checkpointer)
    # return graph, checkpointer
    return workflow


if __name__ == "__main__":
    pass
