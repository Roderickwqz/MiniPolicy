# app/mcp/tools/vector_index.py
from __future__ import annotations

import json
from typing import Any, Dict, List

from app.mcp.contracts import ToolError, ToolResult
from app.mcp.tools.weaviate_client import get_weaviate_client, get_weaviate_vector_store, ensure_weaviate_class

try:
    from llama_index.core import Document
    from llama_index.core.schema import MetadataMode
except ImportError:
    Document = None


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

    if Document is None:
        return ToolResult(
            ok=False,
            tool_name="vector_index_tool",
            error=ToolError(
                code="DEPENDENCY_ERROR",
                message="llama-index is not installed",
                details={"required": "llama-index"},
            ),
        )

    try:
        # Get Weaviate client and vector store
        client = get_weaviate_client()
        if client is None:
            return ToolResult(
                ok=False,
                tool_name="vector_index_tool",
                error=ToolError(
                    code="CONNECTION_ERROR",
                    message="Failed to connect to Weaviate",
                    details={"url": "http://localhost:22006"},
                ),
            )

        # Ensure Weaviate class exists with required properties
        class_name = index_name.replace("::", "_").replace("-", "_")  # Sanitize class name
        properties = [
            {"name": "chunk_id", "dataType": ["string"], "description": "Unique chunk identifier"},
            {"name": "text", "dataType": ["text"], "description": "Chunk text content"},
            {"name": "doc_id", "dataType": ["string"], "description": "Document identifier"},
            {"name": "page", "dataType": ["int"], "description": "Page number"},
            {"name": "hash", "dataType": ["string"], "description": "Chunk hash"},
            {"name": "meta", "dataType": ["text"], "description": "JSON metadata"},
        ]
        ensure_weaviate_class(client, class_name, properties)

        # Get vector store
        vector_store = get_weaviate_vector_store(class_name, text_key="text")
        if vector_store is None:
            return ToolResult(
                ok=False,
                tool_name="vector_index_tool",
                error=ToolError(
                    code="INITIALIZATION_ERROR",
                    message="Failed to initialize WeaviateVectorStore",
                ),
            )

        # Convert chunks to LlamaIndex Documents and upsert
        documents: List[Document] = []
        upserted = 0

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text", "")
            if not chunk_id or not text:
                continue

            meta = chunk.get("meta", {})
            
            # Create Document with metadata
            doc_metadata = {
                "chunk_id": chunk_id,
                "doc_id": meta.get("doc_id", ""),
                "page": meta.get("page", 0),
                "hash": meta.get("hash", ""),
                "meta": json.dumps(meta, ensure_ascii=False),
            }
            
            doc = Document(
                text=text,
                metadata=doc_metadata,
                id_=chunk_id,  # Use chunk_id as document ID for deduplication
            )
            documents.append(doc)
            upserted += 1

        # Batch upsert to Weaviate
        if documents:
            # Add documents to vector store
            # WeaviateVectorStore.add() expects a list of BaseNode or Document
            vector_store.add(documents)

        # Get index size (count documents in class)
        try:
            result = client.query.aggregate(class_name).with_meta_count().do()
            index_size = result.get("data", {}).get("Aggregate", {}).get(class_name, [{}])[0].get("meta", {}).get("count", upserted)
        except Exception:
            index_size = upserted

        return ToolResult(
            ok=True,
            tool_name="vector_index_tool",
            data={"upserted": upserted, "index_size": index_size, "class_name": class_name},
        )

    except Exception as e:
        return ToolResult(
            ok=False,
            tool_name="vector_index_tool",
            error=ToolError(
                code="RUNTIME_ERROR",
                message=f"Failed to index chunks: {str(e)}",
                details={"error_type": type(e).__name__},
            ),
        )
