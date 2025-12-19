from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class LocalFSTool:
    """
    Phase0: 只实现读写 JSON / 文本，所有节点 IO 统一走 gateway 调用它。
    """

    def read_text(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        return {"ok": True, "path": str(p), "text": p.read_text(encoding="utf-8")}

    def write_text(self, path: str, text: str) -> Dict[str, Any]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return {"ok": True, "path": str(p), "bytes": len(text.encode("utf-8"))}

    def write_json(self, path: str, data: Any) -> Dict[str, Any]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True, "path": str(p)}

    def read_json(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        return {"ok": True, "path": str(p), "data": json.loads(p.read_text(encoding="utf-8"))}
