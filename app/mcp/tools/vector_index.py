# app/mcp/tools/vector_index_tool.py
from __future__ import annotations


import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import weaviate
from weaviate.classes.config import Property, DataType
from llama_index.embeddings.openai import OpenAIEmbedding
from weaviate.classes.data import DataObject

from app.mcp.contracts import ToolError, ToolResult
from app.mcp.tools.weaviate_client import get_weaviate_client, ensure_weaviate_collection


def _normalize_collection_name(name: str) -> str:
    parts = re.split(r"[^0-9a-zA-Z]+", (name or "").strip())
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("index_name is empty after normalization")
    class_name = "".join(p[:1].upper() + p[1:] for p in parts)
    if not class_name[0].isalpha():
        class_name = "X" + class_name
    return class_name


def _deterministic_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"minipolicy:{chunk_id}"))


def _chunked(xs: List[Any], n: int) -> List[List[Any]]:
    return [xs[i:i+n] for i in range(0, len(xs), n)]


def vector_index_tool(args: Dict[str, Any]) -> ToolResult:
    tool_name = "vector_index_tool"
    t0 = time.perf_counter()

    index_name = args.get("index_name")
    chunks = args.get("chunks") or []
    args_norm = {"index_name": index_name, "chunk_count": len(chunks) if isinstance(chunks, list) else None}

    if not index_name or not isinstance(index_name, str):
        return ToolResult(False, tool_name, args_norm, error=ToolError("VALIDATION_ERROR", "index_name is required and must be a string"))
    if not isinstance(chunks, list):
        return ToolResult(False, tool_name, args_norm, error=ToolError("VALIDATION_ERROR", "chunks must be a list"))

    try:
        class_name = _normalize_collection_name(index_name)
    except Exception as e:
        return ToolResult(False, tool_name, args_norm, error=ToolError("VALIDATION_ERROR", "invalid index_name", details={"error": str(e)}))

    received = len(chunks)
    valid = 0
    skipped = 0
    skipped_reasons = {"missing_chunk_id": 0, "empty_text": 0}

    # Client-side embedding (LlamaIndex)
    embedding_model = args.get("embedding_model") or "text-embedding-3-small"
    embedder = OpenAIEmbedding(model=embedding_model)

    batch_size = int(args.get("batch_size", 32) or 32)

    client: Optional[weaviate.WeaviateClient] = None
    try:
        client = get_weaviate_client()

        # IMPORTANT: collection must be vectorizer=none so Weaviate will NOT re-vectorize
        properties: List[Property] = [
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="doc_id", data_type=DataType.TEXT),
            Property(name="page", data_type=DataType.INT),
            Property(name="hash", data_type=DataType.TEXT),
            Property(name="meta", data_type=DataType.TEXT),
        ]

        ensure_weaviate_collection(
            client=client,
            class_name=class_name,
            text_key="text",
            properties=properties,
            vectorizer_mode="none",
        )

        collection = client.collections.get(class_name)

        # Build payloads
        rows = []
        texts = []
        for ch in chunks:
            chunk_id = (ch or {}).get("chunk_id")
            text = (ch or {}).get("text", "")
            if not chunk_id:
                skipped += 1
                skipped_reasons["missing_chunk_id"] += 1
                continue
            if not isinstance(text, str) or not text.strip():
                skipped += 1
                skipped_reasons["empty_text"] += 1
                continue

            meta: Dict[str, Any] = (ch or {}).get("meta") or {}
            props = {
                "text": text,
                "chunk_id": str(chunk_id),
                "doc_id": str(meta.get("doc_id", "")),
                "page": int(meta.get("page", 0) or 0),
                "hash": str(meta.get("hash", "")),
                "meta": json.dumps(meta, ensure_ascii=False),
            }
            rows.append((str(chunk_id), props))
            texts.append(text)
            valid += 1

        upserted = 0
        # Embed + insert in batches
        for batch_rows, batch_texts in zip(_chunked(rows, batch_size), _chunked(texts, batch_size)):
            vectors = embedder.get_text_embedding_batch(batch_texts)
            objects = []
            for (chunk_id, props), vec in zip(batch_rows, vectors):
                objects.append(
                    DataObject(
                        uuid=_deterministic_uuid(chunk_id),
                        properties=props,
                        vector=vec,
                    )
                )
            res = collection.data.insert_many(objects)
            # best-effort success counting
            if bool(getattr(res, "has_errors", False)):
                # treat unknown errors as failure
                raise ValueError(f"insert_many has_errors: {getattr(res, 'errors', None)}")
            upserted += len(objects)

        # Index size
        index_size = upserted
        try:
            index_size = collection.aggregate.over_all(total_count=True).total_count
        except Exception:
            pass

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return ToolResult(
            ok=True,
            tool_name=tool_name,
            args=args_norm,
            data={"upserted": upserted, "index_size": int(index_size or 0), "class_name": class_name},
            meta={
                "received": received,
                "valid": valid,
                "skipped": skipped,
                "skipped_reasons": skipped_reasons,
                "elapsed_ms": elapsed_ms,
                "vectorizer": "none",
                "embedding_model": embedding_model,
                "embedding_provider": "llamaindex:OpenAIEmbedding",
                "batch_size": batch_size,
            },
        )

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return ToolResult(
            ok=False,
            tool_name=tool_name,
            args=args_norm,
            meta={"elapsed_ms": elapsed_ms},
            error=ToolError("TOOL_RUNTIME_ERROR", f"Failed to index chunks: {str(e)}", details={"error_type": type(e).__name__}),
        )
    finally:
        if client:
            try:
                client.close()
            except Exception:
                pass
