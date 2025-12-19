# app/mcp/tools/vector_index.py
from __future__ import annotations
from typing import Any, Dict, List
from app.mcp.contracts import ToolError, ToolResult

# P0：用内存 dict 代替向量库
_INDEX: Dict[str, Dict[str, Any]] = {}

def vector_index_tool(args: Dict[str, Any]) -> ToolResult:
    """
    args:
      - index_name: str
      - chunks: list
    returns:
      - upserted: int
    """
    index_name = args["index_name"]
    chunks: List[Dict[str, Any]] = args["chunks"]

    if index_name not in _INDEX:
        _INDEX[index_name] = {"chunks": []}

    try:
        _INDEX[index_name]["chunks"].extend(chunks)
        return ToolResult(ok=True, tool_name="vector_index_tool", data={"upserted": len(chunks)})
    except Exception as e:
        return ToolResult(
            ok=False,
            tool_name="vector_index_tool",
            error=ToolError(code="TOOL_RUNTIME_ERROR", message=str(e)),
        )
