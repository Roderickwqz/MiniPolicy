# app/mcp/contracts.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

# 统一的结构化返回：ToolResult / ToolError（禁止静默失败）

@dataclass
class ToolError:
    code: str               # e.g. "VALIDATION_ERROR", "NOT_ALLOWED", "TOOL_RUNTIME_ERROR"
    message: str            # human readable
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    ok: bool
    tool_name: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[ToolError] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {"ok": self.ok, "tool_name": self.tool_name}
        if self.data is not None:
            payload["data"] = self.data
        if self.error is not None:
            payload["error"] = self.error.to_dict()
        return payload
