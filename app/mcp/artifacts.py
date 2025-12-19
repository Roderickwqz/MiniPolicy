# app/mcp/artifacts.py
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional


_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_run_dir(run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)


def append_artifact(run_dir: str, record: Dict[str, Any]) -> None:
    """
    Append a record into artifacts.json as a JSONL-like array (read-modify-write).
    This is fine for a demo; production would use sqlite or jsonl.
    """
    ensure_run_dir(run_dir)
    path = os.path.join(run_dir, "artifacts.json")

    with _LOCK:
        if not os.path.exists(path):
            payload = {"created_at": _utc_now_iso(), "records": []}
        else:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

        record = {
            "ts": _utc_now_iso(),
            **record,
        }
        payload["records"].append(record)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def append_audit_log(run_dir: str, line: str) -> None:
    ensure_run_dir(run_dir)
    path = os.path.join(run_dir, "audit.log")
    with _LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
