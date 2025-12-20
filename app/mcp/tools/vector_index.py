# app/mcp/tools/vector_index.py
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

from app.mcp.contracts import ToolError, ToolResult

# 简易向量索引：Bag-of-words + cosine
_INDEX: Dict[str, Dict[str, Any]] = {}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _embed(text: str) -> Tuple[Counter, float]:
    tokens = _tokenize(text)
    counts = Counter(tokens)
    norm = math.sqrt(sum(v * v for v in counts.values()))
    return counts, norm


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

    store = _INDEX.setdefault(index_name, {"chunks": {}, "embeddings": {}})

    upserted = 0
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text", "")
        if not chunk_id or not text:
            continue
        embedding, norm = _embed(text)
        store["chunks"][chunk_id] = chunk
        store["embeddings"][chunk_id] = {"vector": embedding, "norm": norm}
        upserted += 1

    return ToolResult(
        ok=True,
        tool_name="vector_index_tool",
        data={"upserted": upserted, "index_size": len(store["chunks"])},
    )
