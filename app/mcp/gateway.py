from __future__ import annotations

import time
from typing import Any, Dict

from app.core.envelope import Envelope
from app.mcp.tools.local_fs import LocalFSTool


class MCPGateway:
    def __init__(self) -> None:
        self.fs = LocalFSTool()

    def call_tool(
        self,
        *,
        run_id: str,
        node_id: str,
        step_id: str,
        graph_path_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        chunk_ids: list[str] | None = None,
    ) -> tuple[Envelope, Envelope]:
        """
        返回 (request_envelope, result_envelope)
        """
        chunk_ids = chunk_ids or []
        ts = int(time.time() * 1000)

        req = Envelope(
            envelope_type="tool.request",
            run_id=run_id,
            node_id=node_id,
            step_id=step_id,
            ts_ms=ts,
            graph_path_id=graph_path_id,
            chunk_ids=chunk_ids,
            payload={"tool": tool_name, "args": tool_args},
        )

        # 调用本地工具（Phase0 先同步）
        try:
            if tool_name == "fs.read_text":
                out = self.fs.read_text(**tool_args)
            elif tool_name == "fs.write_text":
                out = self.fs.write_text(**tool_args)
            elif tool_name == "fs.write_json":
                out = self.fs.write_json(**tool_args)
            elif tool_name == "fs.read_json":
                out = self.fs.read_json(**tool_args)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            res = Envelope(
                envelope_type="tool.result",
                run_id=run_id,
                node_id=node_id,
                step_id=step_id,
                ts_ms=int(time.time() * 1000),
                graph_path_id=graph_path_id,
                chunk_ids=chunk_ids,
                payload={"tool": tool_name, "output": out},
            )
            return req, res

        except Exception as e:
            res = Envelope(
                envelope_type="tool.result",
                run_id=run_id,
                node_id=node_id,
                step_id=step_id,
                ts_ms=int(time.time() * 1000),
                graph_path_id=graph_path_id,
                chunk_ids=chunk_ids,
                payload={"tool": tool_name, "output": None},
                error={"type": type(e).__name__, "message": str(e)},
            )
            return req, res
