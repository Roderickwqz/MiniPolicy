# app/mcp/tools/semantic_retrieve.py
from __future__ import annotations
from typing import Any, Dict, List
from app.mcp.contracts import ToolError, ToolResult
from app.mcp.tools.vector_index import _INDEX  # P0 demo共享内存索引

def semantic_retrieve_tool(args: Dict[str, Any]) -> ToolResult:
    """
    args:
      - index_name: str
      - query: str
      - top_k: int
    returns:
      - matches: [{chunk_id, text, score, meta}]
    """
    index_name = args["index_name"]
    query = args["query"]
    top_k = args["top_k"]

    if index_name not in _INDEX:
        return ToolResult(
            ok=False,
            tool_name="semantic_retrieve_tool",
            error=ToolError(code="NOT_FOUND", message="Index not found", details={"index_name": index_name}),
        )

    # P0：不用 embedding，做“包含关系”打分，保证可运行
    chunks: List[Dict[str, Any]] = _INDEX[index_name]["chunks"]
    scored = []
    for c in chunks:
        text = c.get("text", "")
        score = 1.0 if query and query in text else 0.0
        scored.append({**c, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return ToolResult(ok=True, tool_name="semantic_retrieve_tool", data={"matches": scored[:top_k]})
