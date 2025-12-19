# app/mcp/tools/graph_store.py
from __future__ import annotations
from typing import Any, Dict, List
from app.mcp.contracts import ToolError, ToolResult

_GRAPH: Dict[str, Any] = {"nodes": [], "edges": []}

def graph_upsert_tool(args: Dict[str, Any]) -> ToolResult:
    """
    args:
      - nodes: list
      - edges: list
    returns:
      - nodes_total, edges_total
    """
    nodes = args["nodes"]
    edges = args["edges"]
    try:
        _GRAPH["nodes"].extend(nodes)
        _GRAPH["edges"].extend(edges)
        return ToolResult(
            ok=True,
            tool_name="graph_upsert_tool",
            data={"nodes_total": len(_GRAPH["nodes"]), "edges_total": len(_GRAPH["edges"])},
        )
    except Exception as e:
        return ToolResult(ok=False, tool_name="graph_upsert_tool", error=ToolError(code="TOOL_RUNTIME_ERROR", message=str(e)))


def graph_query_tool(args: Dict[str, Any]) -> ToolResult:
    """
    args:
      - query: str
    returns:
      - graph_path_id: str
    """
    query = args["query"]
    # P0：不做真实 query，返回一个可追踪 id
    graph_path_id = f"graph_path::{abs(hash(query))}"
    return ToolResult(ok=True, tool_name="graph_query_tool", data={"graph_path_id": graph_path_id})
