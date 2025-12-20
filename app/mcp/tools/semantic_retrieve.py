# app/mcp/tools/semantic_retrieve.py
from __future__ import annotations

import math
from typing import Any, Dict, List

from app.mcp.contracts import ToolError, ToolResult
from app.mcp.tools.vector_index import _INDEX, _tokenize


def _cosine(a: Dict[str, int], b: Dict[str, int], norm_a: float, norm_b: float) -> float:
    if norm_a == 0 or norm_b == 0:
        return 0.0
    dot = 0.0
    for token, weight in a.items():
        dot += weight * b.get(token, 0)
    return dot / (norm_a * norm_b) if dot else 0.0


def semantic_retrieve_tool(args: Dict[str, Any]) -> ToolResult:
    """
    args:
      - index_name: str
      - query: str
      - top_k: int
    returns:
      - matches: [{chunk_id, text, score, meta}]
    """
    index_name = args["index_name"]
    query = args["query"]
    top_k = max(1, int(args["top_k"]))

    store = _INDEX.get(index_name)
    if not store:
        return ToolResult(
            ok=False,
            tool_name="semantic_retrieve_tool",
            error=ToolError(code="NOT_FOUND", message="Index not found", details={"index_name": index_name}),
        )

    query_tokens = _tokenize(query)
    query_counts: Dict[str, int] = {}
    for token in query_tokens:
        query_counts[token] = query_counts.get(token, 0) + 1
    norm_q = math.sqrt(sum(v * v for v in query_counts.values()))

    matches: List[Dict[str, Any]] = []
    for chunk_id, chunk in store["chunks"].items():
        embed_info = store["embeddings"].get(chunk_id)
        if not embed_info:
            continue
        score = _cosine(query_counts, embed_info["vector"], norm_q, embed_info["norm"])
        matches.append(
            {
                "chunk_id": chunk_id,
                "text": chunk.get("text", ""),
                "score": score,
                "meta": chunk.get("meta", {}),
            }
        )

    matches.sort(key=lambda x: x["score"], reverse=True)
    return ToolResult(
        ok=True,
        tool_name="semantic_retrieve_tool",
        data={"matches": matches[:top_k], "query": query},
    )
