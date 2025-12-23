# app/mcp/tools/semantic_retrieve.py
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.schema import NodeWithScore

from app.mcp.contracts import ToolError, ToolResult
from app.mcp.tools.weaviate_client import get_weaviate_client, get_weaviate_vector_store
from app.mcp.tools.vector_index import normalize_collection_name 


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def semantic_retrieve_tool(args: Dict[str, Any]) -> ToolResult:
    """
    Query Weaviate vector database for semantically similar chunks.

    Required args:
      - index_name: Weaviate class/collection name
      - query:      natural language query string

    Optional args:
      - top_k:      int (default=1, min=1)

    Returns ToolResult:
      ok=True:
        data={
          "matches": [{"chunk_id","text","score","meta"}...],
          "query": str,
          "index_name": str,
          "class_name": str,
          "top_k": int
        }
        meta={...}
      ok=False:
        error={code,message,details}
        meta={...}
    """
    t0 = time.time()
    tool_name = "semantic_retrieve_tool"

    # ---------- 1) Normalize & validate inputs ----------
    index_name = (args.get("index_name") or "").strip()
    query = (args.get("query") or "").strip()
    top_k = max(1, _safe_int(args.get("top_k", 1), 1))

    normalized_args = {
        "index_name": index_name,
        "query": query,
        "top_k": top_k,
    }

    if not index_name or not query:
        return ToolResult(
            ok=False,
            tool_name=tool_name,
            args=normalized_args,
            error=ToolError(
                code="VALIDATION_ERROR",
                message="Missing required arguments: index_name and query are required.",
                details={"required": ["index_name", "query"]},
            ),
            meta={"timing_ms": int((time.time() - t0) * 1000)},
        )

    if VectorStoreIndex is None or QueryBundle is None:
        return ToolResult(
            ok=False,
            tool_name=tool_name,
            args=normalized_args,
            error=ToolError(
                code="DEPENDENCY_ERROR",
                message="llama-index is not installed; cannot build VectorStoreIndex.",
                details={"required": "llama-index"},
            ),
            meta={"timing_ms": int((time.time() - t0) * 1000)},
        )

    class_name = normalize_collection_name(index_name)

    client = None
    vector_store = None
    warnings: List[str] = []

    try:
        # ---------- 2) Connect Weaviate ----------
        client = get_weaviate_client()
        if client is None:
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                args={**normalized_args, "class_name": class_name},
                error=ToolError(
                    code="CONNECTION_ERROR",
                    message="Failed to connect to Weaviate client.",
                    details={"url": "http://localhost:22006"},
                ),
                meta={"timing_ms": int((time.time() - t0) * 1000)},
            )

        weaviate_api = "v4" if hasattr(client, "collections") else "v3"

        # ---------- 3) Optional: check index existence ----------
        # 注意：schema 检查失败不应静默吞掉；这里转为 warning（不影响继续尝试）
        try:
            if weaviate_api == "v4":
                # v4: collections.get 不存在会抛异常
                client.collections.get(class_name)
            else:
                schema = client.schema.get()
                existing_classes = [c.get("class") for c in schema.get("classes", [])]
                if class_name not in existing_classes:
                    return ToolResult(
                        ok=False,
                        tool_name=tool_name,
                        args={**normalized_args, "class_name": class_name},
                        error=ToolError(
                            code="NOT_FOUND",
                            message="Index not found in Weaviate schema.",
                            details={"index_name": index_name, "class_name": class_name},
                        ),
                        meta={
                            "weaviate_api": weaviate_api,
                            "timing_ms": int((time.time() - t0) * 1000),
                        },
                    )
        except ToolResult:
            raise
        except Exception as e:
            warnings.append(f"schema_check_failed:{type(e).__name__}:{str(e)[:200]}")

        # ---------- 4) Build vector store + index ----------
        vector_store = get_weaviate_vector_store(client, class_name, text_key="text")
        if vector_store is None:
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                args={**normalized_args, "class_name": class_name},
                error=ToolError(
                    code="INITIALIZATION_ERROR",
                    message="Failed to initialize WeaviateVectorStore.",
                    details={"class_name": class_name},
                ),
                meta={
                    "weaviate_api": weaviate_api,
                    "timing_ms": int((time.time() - t0) * 1000),
                    "warnings": warnings or None,
                },
            )

        index = VectorStoreIndex.from_vector_store(vector_store)
        retriever = index.as_retriever(similarity_top_k=top_k)

        # ---------- 5) Retrieve ----------
        query_bundle = QueryBundle(query_str=query)

        try:
            nodes_with_scores: List[NodeWithScore] = retriever.retrieve(query_bundle)
        except Exception as e:
            # 这里按“禁止静默失败”处理：检索失败直接返回 ok=False
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                args={**normalized_args, "class_name": class_name},
                error=ToolError(
                    code="TOOL_RUNTIME_ERROR",
                    message="Vector retrieval failed.",
                    details={
                        "error_type": type(e).__name__,
                        "error": str(e),
                    },
                ),
                meta={
                    "weaviate_api": weaviate_api,
                    "timing_ms": int((time.time() - t0) * 1000),
                    "warnings": warnings or None,
                },
            )

            # 如果你更想“失败也 ok=True 但给 warning + 空结果”，改成下面这种：
            # warnings.append(f"retrieve_failed:{type(e).__name__}:{str(e)[:200]}")
            # nodes_with_scores = []

        # ---------- 6) Map results ----------
        matches: List[Dict[str, Any]] = []
        for nws in nodes_with_scores or []:
            node = nws.node
            score = float(nws.score or 0.0)

            metadata = node.metadata or {}
            chunk_id = metadata.get("chunk_id") or getattr(node, "node_id", None) or ""
            text = node.get_content() or ""

            meta_json = metadata.get("meta", None)
            meta: Dict[str, Any]
            try:
                if isinstance(meta_json, str):
                    meta = json.loads(meta_json)
                elif isinstance(meta_json, dict):
                    meta = meta_json
                else:
                    meta = {}
            except Exception:
                meta = {}

            # 补齐常见字段（兼容旧数据）
            if "doc_id" not in meta:
                meta["doc_id"] = metadata.get("doc_id", "")
            if "page" not in meta:
                meta["page"] = metadata.get("page", 0)
            if "hash" not in meta:
                meta["hash"] = metadata.get("hash", "")

            matches.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": score,
                    "meta": meta,
                }
            )

        matches.sort(key=lambda x: x["score"], reverse=True)

        return ToolResult(
            ok=True,
            tool_name=tool_name,
            args={**normalized_args, "class_name": class_name},
            data={
                "matches": matches,
                "query": query,
                "index_name": index_name,
                "class_name": class_name,
                "top_k": top_k,
            },
            meta={
                "weaviate_api": weaviate_api,
                "timing_ms": int((time.time() - t0) * 1000),
                "warnings": warnings or None,
                "match_count": len(matches),
            },
        )

    except Exception as e:
        return ToolResult(
            ok=False,
            tool_name=tool_name,
            args={**normalized_args, "class_name": class_name},
            error=ToolError(
                code="TOOL_RUNTIME_ERROR",
                message="Unexpected error in semantic_retrieve_tool.",
                details={"error_type": type(e).__name__, "error": str(e)},
            ),
            meta={
                "timing_ms": int((time.time() - t0) * 1000),
                "warnings": warnings or None,
            },
        )
    finally:
        # v4 client 需要 close；vector_store 可能持有 client
        if vector_store is not None and hasattr(vector_store, "_client"):
            try:
                vector_store._client.close()
            except Exception:
                pass
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
