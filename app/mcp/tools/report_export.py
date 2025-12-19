# app/mcp/tools/report_export.py
from __future__ import annotations
import json
import os
from typing import Any, Dict
from app.mcp.contracts import ToolError, ToolResult

def report_export_tool(args: Dict[str, Any]) -> ToolResult:
    """
    args:
      - run_dir: str
      - format: "md" | "json"
      - content: object (md string or dict)
      - filename: str
    returns:
      - path: str
    """
    run_dir = args["run_dir"]
    fmt = args["format"]
    content = args["content"]
    filename = args["filename"]

    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, filename)

    try:
        if fmt == "md":
            if not isinstance(content, str):
                return ToolResult(ok=False, tool_name="report_export_tool",
                                  error=ToolError(code="VALIDATION_ERROR", message="md content must be string"))
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        elif fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
        else:
            return ToolResult(ok=False, tool_name="report_export_tool",
                              error=ToolError(code="VALIDATION_ERROR", message=f"Unsupported format: {fmt}"))

        return ToolResult(ok=True, tool_name="report_export_tool", data={"path": path})

    except Exception as e:
        return ToolResult(ok=False, tool_name="report_export_tool",
                          error=ToolError(code="TOOL_RUNTIME_ERROR", message=str(e)))
