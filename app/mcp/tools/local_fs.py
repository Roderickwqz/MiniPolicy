from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

# 项目根目录（假设 LocalFSTool 在 app/mcp/tools/ 下）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 允许的沙箱目录
_ALLOWED_ROOTS = [
    _PROJECT_ROOT / "app" / "artifacts",
    _PROJECT_ROOT / "app" / "runs",
    _PROJECT_ROOT / "docs",
]


class LocalFSTool:
    """
    Phase0/1: 本地文件工具。节点不直接 IO，统一走 gateway 调用它。
    """

    def _resolve(self, path: str) -> Path:
        """
        将路径解析到项目根目录下的指定沙箱，并进行安全检查。
        - 禁止绝对路径（如 /etc/passwd）
        - 禁止路径逃逸（如 ../../）
        - 只允许在允许的沙箱目录内操作
        """
        p = Path(path)

        # 检查是否为绝对路径
        if p.is_absolute():
            raise ValueError(f"Absolute path not allowed: {path}")

        # 解析到项目根目录
        resolved = (_PROJECT_ROOT / p).resolve()

        # 检查是否尝试逃逸（通过检查是否在项目根目录外）
        if not resolved.is_relative_to(_PROJECT_ROOT):
            raise ValueError(f"Path escape attempt not allowed: {path}")

        # 检查是否在允许的沙箱目录内
        allowed = False
        for allowed_root in _ALLOWED_ROOTS:
            try:
                resolved.relative_to(allowed_root)
                allowed = True
                break
            except ValueError:
                continue

        if not allowed:
            raise ValueError(f"Path not in allowed sandbox: {path}")

        return resolved

    def read_text(self, path: str) -> Dict[str, Any]:
        p = self._resolve(path)
        return {"ok": True, "path": str(p), "text": p.read_text(encoding="utf-8")}

    def write_text(self, path: str, text: str) -> Dict[str, Any]:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return {"ok": True, "path": str(p), "bytes": len(text.encode("utf-8"))}

    def append_text(self, path: str, text: str) -> Dict[str, Any]:
        """
        追加写文本（用于 audit.log / jsonl 等）。
        """
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(text)
        return {"ok": True, "path": str(p), "bytes": len(text.encode("utf-8"))}

    def write_json(self, path: str, data: Any) -> Dict[str, Any]:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True, "path": str(p)}

    def read_json(self, path: str) -> Dict[str, Any]:
        p = self._resolve(path)
        return {"ok": True, "path": str(p), "data": json.loads(p.read_text(encoding="utf-8"))}

    def append_json_record(self, path: str, record: Any) -> Dict[str, Any]:
        """
        将 record 追加到 artifacts.json 的 records[] 中。
        仍是读-改-写，但把语义固定下来，避免 nodes 覆盖 artifacts。
        """
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if p.exists():
            payload = json.loads(p.read_text(encoding="utf-8"))
        else:
            payload = {}

        records = payload.get("records", [])
        if not isinstance(records, list):
            records = []

        records.append(record)
        payload["records"] = records

        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True, "path": str(p), "records": len(records)}
