# app/mcp/gateway.py
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple

from app.core.envelope import Envelope
from app.mcp.registry import ToolSpec, build_registry
from app.mcp.tools.local_fs import LocalFSTool


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _validate_args(schema: Dict[str, Any], args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    required = schema.get("required", [])
    props = schema.get("properties", {})

    for k in required:
        if k not in args:
            return False, f"Missing required arg: {k}"

    for k, v in args.items():
        if k not in props:
            return False, f"Unknown arg: {k}"
        t = props[k].get("type")
        if t == "string" and not isinstance(v, str):
            return False, f"Arg '{k}' must be string"
        if t == "integer" and not isinstance(v, int):
            return False, f"Arg '{k}' must be integer"
        if t == "number" and not isinstance(v, (int, float)):
            return False, f"Arg '{k}' must be number"
        if t == "boolean" and not isinstance(v, bool):
            return False, f"Arg '{k}' must be boolean"
        if t == "object" and not isinstance(v, dict):
            return False, f"Arg '{k}' must be object"
        if t == "array" and not isinstance(v, list):
            return False, f"Arg '{k}' must be array"

    return True, None


class MCPGateway:
    def __init__(self) -> None:
        self.fs = LocalFSTool()
        self.registry: Dict[str, ToolSpec] = build_registry(self.fs)

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
        并在 gateway 内部强制写审计记录：
          - app/artifacts/<run_id>/artifacts.json 追加 tool_call record
          - app/artifacts/<run_id>/audit.log 追加一行
        """
        chunk_ids = chunk_ids or []
        t0 = time.perf_counter()
        ts = _now_ms()
        args_hash = _sha256(_stable_json(tool_args))

        req = Envelope(
            envelope_type="tool.request",
            run_id=run_id,
            node_id=node_id,
            step_id=step_id,
            ts_ms=ts,
            graph_path_id=graph_path_id,
            chunk_ids=chunk_ids,
            payload={"tool": tool_name, "args": tool_args, "args_hash": args_hash},
        )

        spec = self.registry.get(tool_name)
        if spec is None or not spec.allow:
            res = Envelope(
                envelope_type="tool.result",
                run_id=run_id,
                node_id=node_id,
                step_id=step_id,
                ts_ms=_now_ms(),
                graph_path_id=graph_path_id,
                chunk_ids=chunk_ids,
                payload={"tool": tool_name, "output": None},
                error={"code": "NOT_ALLOWED", "type": "PermissionError", "message": f"Tool not allowed: {tool_name}"},
            )
            self._audit(run_id, req, res, args_hash=args_hash, output_hash=None, latency_ms=(time.perf_counter() - t0) * 1000, status="DENY")
            return req, res

        ok, err = _validate_args(spec.schema, tool_args)
        if not ok:
            res = Envelope(
                envelope_type="tool.result",
                run_id=run_id,
                node_id=node_id,
                step_id=step_id,
                ts_ms=_now_ms(),
                graph_path_id=graph_path_id,
                chunk_ids=chunk_ids,
                payload={"tool": tool_name, "output": None},
                error={"code": "VALIDATION_ERROR", "type": "ValueError", "message": err, "details": {"schema": spec.schema}},
            )
            self._audit(run_id, req, res, args_hash=args_hash, output_hash=None, latency_ms=(time.perf_counter() - t0) * 1000, status="INVALID_ARGS")
            return req, res

        try:
            out = spec.handler(tool_args)
            latency_ms = (time.perf_counter() - t0) * 1000
            output_hash = _sha256(_stable_json(out))

            res = Envelope(
                envelope_type="tool.result",
                run_id=run_id,
                node_id=node_id,
                step_id=step_id,
                ts_ms=_now_ms(),
                graph_path_id=graph_path_id,
                chunk_ids=chunk_ids,
                payload={"tool": tool_name, "output": out, "output_hash": output_hash},
            )
            self._audit(run_id, req, res, args_hash=args_hash, output_hash=output_hash, latency_ms=latency_ms, status="OK")
            return req, res

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            res = Envelope(
                envelope_type="tool.result",
                run_id=run_id,
                node_id=node_id,
                step_id=step_id,
                ts_ms=_now_ms(),
                graph_path_id=graph_path_id,
                chunk_ids=chunk_ids,
                payload={"tool": tool_name, "output": None},
                error={"code": "TOOL_RUNTIME_ERROR", "type": type(e).__name__, "message": str(e)},
            )
            self._audit(run_id, req, res, args_hash=args_hash, output_hash=None, latency_ms=latency_ms, status="EXCEPTION")
            return req, res

    def _audit(
        self,
        run_id: str,
        req: Envelope,
        res: Envelope,
        *,
        args_hash: str,
        output_hash: Optional[str],
        latency_ms: float,
        status: str,
    ) -> None:
        run_dir = f"app/artifacts/{run_id}"
        artifacts_path = f"{run_dir}/artifacts.json"
        audit_path = f"{run_dir}/audit.log"

        # tool_call record 追加到 artifacts.json
        record = {
            "type": "tool_call",
            "tool_name": req.payload.get("tool"),
            "args_hash": args_hash,
            "output_hash": output_hash,
            "latency_ms": round(latency_ms, 3),
            "status": status,
            "request": req.model_dump() if hasattr(req, "model_dump") else req.__dict__,
            "result": res.model_dump() if hasattr(res, "model_dump") else res.__dict__,
        }
        self.fs.append_json_record(path=artifacts_path, record=record)

        # audit.log 追加一行（JSON）
        line = _stable_json(
            {
                "tool_name": req.payload.get("tool"),
                "args_hash": args_hash,
                "output_hash": output_hash,
                "latency_ms": round(latency_ms, 3),
                "status": status,
            }
        )
        self.fs.append_text(path=audit_path, text=line + "\n")
