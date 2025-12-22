# app/mcp/tools/vector_index.py
from __future__ import annotations

import json
from typing import Any, Dict, List
import weaviate
import re

from app.mcp.contracts import ToolError, ToolResult
from app.mcp.tools.weaviate_client import get_weaviate_client, get_weaviate_vector_store, ensure_weaviate_class

try:
    from llama_index.core import Document  # 原始文档的标准载体，用来承载整篇内容 + 元数据
except ImportError:
    Document = None


def normalize_class_name(name: str) -> str:
    parts = re.split(r"[^0-9a-zA-Z]+", name.strip())
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("Index name is empty after normalization")
    class_name = "".join(p[:1].upper() + p[1:] for p in parts)
    if not class_name[0].isalpha():
        class_name = "X" + class_name
    return class_name


def vector_index_tool(args: Dict[str, Any]) -> ToolResult:
    """
    Store chunks from PDF ingest into Weaviate vector database.
    
    This tool is part of the RAG pipeline:
    - PDF → pdf_ingest_tool → chunks (in-memory)
    - chunks → vector_index_tool → Weaviate (vector database with embeddings)
    - Weaviate → semantic_retrieve_tool → top_k chunks (semantic retrieval)
    
    This function directly uses WeaviateVectorStore.add() to upsert documents.
    It does NOT create a query engine or retriever - those are created in
    semantic_retrieve_tool when querying is needed.
    
    Note: We only use VectorStoreIndex (not SummaryIndex) because:
    - VectorStoreIndex: For semantic search/retrieval (our use case)
    - SummaryIndex: For generating summaries of entire dataset (not needed here)
    
    Args:
        index_name: Weaviate class name (will be sanitized for valid class name)
        chunks: List of chunk dictionaries from pdf_ingest_tool, each containing:
            - chunk_id: Unique identifier
            - text: Chunk text content
            - meta: Metadata dict with doc_id, page, hash, etc.
    
    Returns:
        ToolResult with:
            - ok: bool
            - data: {
                "upserted": int,      # Number of chunks successfully stored
                "index_size": int,    # Total chunks in the index
                "class_name": str     # Actual Weaviate class name used
              }
            - error: ToolError if failed
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

    client: weaviate.Client = None
    vector_store = None
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
        # Normalize class name to meet Weaviate requirements (must start with capital letter)
        class_name = normalize_class_name(index_name)
        properties = [
            {"name": "chunk_id", "dataType": ["string"], "description": "Unique chunk identifier"},
            {"name": "text", "dataType": ["text"], "description": "Chunk text content"},
            {"name": "doc_id", "dataType": ["string"], "description": "Document identifier"},
            {"name": "page", "dataType": ["int"], "description": "Page number"},
            {"name": "hash", "dataType": ["string"], "description": "Chunk hash"},
            {"name": "meta", "dataType": ["text"], "description": "JSON metadata"},
        ]
        ensure_weaviate_class(client, class_name, properties)

        # Get vector store (this creates another client internally)
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

            meta: Dict[str, Any] = chunk.get("meta", {})
            
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
        # Use the client from vector_store if available, otherwise use our client
        index_size = upserted
        try:
            # Get the client from vector_store (it has its own client)
            store_client = vector_store._client if hasattr(vector_store, "_client") else client
            
            # Try v4 API first
            if hasattr(store_client, "collections"):
                try:
                    collection = store_client.collections.get(class_name)
                    index_size = collection.aggregate.over_all(total_count=True).total_count
                except Exception:
                    # Collection might not be ready yet, use upserted count
                    index_size = upserted
            elif hasattr(store_client, "query"):
                # Fallback to v3 API
                try:
                    result = store_client.query.aggregate(class_name).with_meta_count().do()
                    index_size = result.get("data", {}).get("Aggregate", {}).get(class_name, [{}])[0].get("meta", {}).get("count", upserted)
                except Exception:
                    index_size = upserted
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
    finally:
        # Close all Weaviate client connections (v4 API requires explicit close)
        # Close vector_store's client if it has one
        if vector_store and hasattr(vector_store, "_client"):
            try:
                vector_store._client.close()
            except Exception:
                pass
        # Close our client
        if client:
            try:
                client.close()
            except Exception:
                pass
