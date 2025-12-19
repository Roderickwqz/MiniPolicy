# app/mcp/tools/pdf_ingest.py
from __future__ import annotations

from typing import Any, Dict, List
from app.mcp.contracts import ToolError, ToolResult

def pdf_ingest_tool(args: Dict[str, Any]) -> ToolResult:
    """
    args:
      - pdf_path: str
      - chunk_size: int
      - overlap: int
    returns:
      - chunks: [{chunk_id, text, meta}]
    """
    pdf_path = args["pdf_path"]
    chunk_size = args["chunk_size"]
    overlap = args["overlap"]

    try:
        # P0：尽量不引入重依赖。若你装了 pypdf，可替换这里。
        with open(pdf_path, "rb") as f:
            raw = f.read()

        # demo：不做真正 PDF 解析，先用 bytes 长度占位，避免 silent fail
        text = f"[P0-DEMO] PDF bytes={len(raw)} path={pdf_path}"

        chunks: List[Dict[str, Any]] = []
        i = 0
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end]
            chunks.append(
                {
                    "chunk_id": f"c{i}",
                    "text": chunk_text,
                    "meta": {"source": pdf_path, "start": start, "end": end},
                }
            )
            i += 1
            start = max(end - overlap, end)

        return ToolResult(ok=True, tool_name="pdf_ingest_tool", data={"chunks": chunks})

    except FileNotFoundError:
        return ToolResult(
            ok=False,
            tool_name="pdf_ingest_tool",
            error=ToolError(code="NOT_FOUND", message="PDF not found", details={"pdf_path": pdf_path}),
        )
