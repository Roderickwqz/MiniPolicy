# app/mcp/gateway.py
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple

from app.core.envelope import Envelope
from app.mcp.registry import ToolSpec, build_registry
from app.mcp.tools.local_fs import LocalFSTool
from app.mcp.artifacts import append_artifact, append_audit_log
from app.mcp.contracts import ToolResult

# Import centralized configuration for validation
from app.mcp.config import get_config


def _stable_json(obj: Any) -> str:
    """
    这个函数的作用是：

    - 将任意 Python 对象转换为 JSON 字符串
    - `ensure_ascii=False`：允许输出中文等非 ASCII 字符，而不是转义为 Unicode
    - `sort_keys=True`：对字典的键进行排序，确保相同内容总是生成相同的字符串
    - `default=str`：对于无法直接序列化的对象（如 datetime、自定义对象等），使用 `str()` 函数将其转换为字符串

    __用途__：生成稳定、可重复的 JSON 字符串表示，用于后续的哈希计算。

    """
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def _sha256(s: str) -> str:
    """
    这个函数的作用是：

    - 计算输入字符串的 SHA-256 哈希值
    - 将字符串编码为 UTF-8 字节
    - 使用 hashlib.sha256 计算哈希
    - 返回十六进制格式的哈希字符串（64个字符）

    __用途__：在代码中用于：

    1. 计算工具参数的哈希值（`args_hash`）
    2. 计算工具输出结果的哈希值（`output_hash`）
    3. 用于审计日志和记录，确保数据完整性
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _validate_args(schema: Dict[str, Any], args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    这个 `_validate_args` 函数是一&#x4E2A;__&#x53C2;数验证器__，用于在调用工具之前验证传入的参数是否符合预期的模式（schema）。

    ## 主要功能
    这个函数接收两个参数：
    - `schema`: 定义了工具期望的参数结构和类型
    - `args`: 实际传入的参数

    返回一个元组 `(bool, Optional[str])`，其中：
    - 第一个值表示验证是否通过（True/False）
    - 第二个值是错误信息（如果验证失败）
    """
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
    def __init__(self, registry: Optional[Dict[str, ToolSpec]] = None) -> None:
        self.fs = LocalFSTool()
        self.registry: Dict[str, ToolSpec] = registry or build_registry(self.fs)

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

        # __req（请求信封）__：记录工具调用的请求信息，包括工具名称、参数、时间戳等
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
            # - __res（结果信封）__：记录工具执行的结果，包括输出、错误信息、状态等
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
            serialized_out = out.to_dict() if isinstance(out, ToolResult) else out
            latency_ms = (time.perf_counter() - t0) * 1000
            output_hash = _sha256(_stable_json(serialized_out))

            res = Envelope(
                envelope_type="tool.result",
                run_id=run_id,
                node_id=node_id,
                step_id=step_id,
                ts_ms=_now_ms(),
                graph_path_id=graph_path_id,
                chunk_ids=chunk_ids,
                payload={"tool": tool_name, "output": serialized_out, "output_hash": output_hash},
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
        append_artifact(run_dir, record)

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
        append_audit_log(run_dir, line)


def build_gateway() -> MCPGateway:
    return MCPGateway()
