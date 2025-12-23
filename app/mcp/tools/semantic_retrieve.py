# app/mcp/tools/semantic_retrieve.py
from __future__ import annotations

import json
from typing import Any, Dict, List

from app.mcp.contracts import ToolError, ToolResult
from app.mcp.tools.weaviate_client import get_weaviate_client, get_weaviate_vector_store
# from app.mcp.tools.vector_index import normalize_class_name

try:
    from llama_index.core import VectorStoreIndex, QueryBundle
    from llama_index.core.schema import NodeWithScore
except ImportError:
    VectorStoreIndex = None
    QueryBundle = None
    NodeWithScore = None


def semantic_retrieve_tool(args: Dict[str, Any]) -> ToolResult:
    """
    Query Weaviate vector database for semantically similar chunks.
    
    This tool creates a VectorStoreIndex from the existing Weaviate vector store,
    then uses a retriever to find top_k most similar chunks to the query.
    
    The workflow is:
    1. Get WeaviateVectorStore (connection to Weaviate)
    2. Create VectorStoreIndex.from_vector_store() (wrapper for querying)
    3. Create retriever with index.as_retriever() (retrieval logic)
    4. Retrieve top_k nodes with retriever.retrieve() (actual query)
    
    Note: We use VectorStoreIndex (not SummaryIndex) because:
    - VectorStoreIndex: For semantic similarity search (our use case)
    - SummaryIndex: For generating summaries of entire dataset (not needed)
    
    Args:
        index_name: Weaviate class name (must match the one used in vector_index_tool)
        query: Natural language query string
        top_k: Number of top results to return (default: 1, minimum: 1)
    
    Returns:
        ToolResult with:
            - ok: bool
            - data: {
                "matches": [{
                    "chunk_id": str,    # Unique chunk identifier
                    "text": str,         # Chunk text content
                    "score": float,      # Similarity score (0.0-1.0)
                    "meta": dict        # Metadata (doc_id, page, hash, etc.)
                }],
                "query": str            # Original query string
              }
            - error: ToolError if failed
    """
    index_name = args["index_name"]
    query = args["query"]
    top_k = max(1, int(args["top_k"]))

    if VectorStoreIndex is None:
        return ToolResult(
            ok=False,
            tool_name="semantic_retrieve_tool",
            error=ToolError(
                code="DEPENDENCY_ERROR",
                message="llama-index is not installed",
                details={"required": "llama-index"},
            ),
        )

    client = None
    vector_store = None
    try:
        # Normalize class name to meet Weaviate requirements (must start with capital letter)
        # Use the same normalization function as vector_index_tool to ensure consistency
        # class_name = normalize_class_name(index_name)
        
        # Get Weaviate client and check if class exists
        client = get_weaviate_client()
        if client is None:
            return ToolResult(
                ok=False,
                tool_name="semantic_retrieve_tool",
                error=ToolError(
                    code="CONNECTION_ERROR",
                    message="Failed to connect to Weaviate",
                    details={"url": "http://localhost:22006"},
                ),
            )

        # Check if class exists
        try:
            if hasattr(client, "collections"):
                # v4 API
                try:
                    collection = client.collections.get(class_name)
                    # If we can get the collection, it exists
                except Exception:
                    return ToolResult(
                        ok=False,
                        tool_name="semantic_retrieve_tool",
                        error=ToolError(
                            code="NOT_FOUND",
                            message="Index not found",
                            details={"index_name": index_name, "class_name": class_name},
                        ),
                    )
            else:
                # v3 API
                schema = client.schema.get()
                existing_classes = [cls.get("class") for cls in schema.get("classes", [])]
                if class_name not in existing_classes:
                    return ToolResult(
                        ok=False,
                        tool_name="semantic_retrieve_tool",
                        error=ToolError(
                            code="NOT_FOUND",
                            message="Index not found",
                            details={"index_name": index_name, "class_name": class_name},
                        ),
                    )
        except Exception:
            # If we can't check schema, try to proceed anyway
            pass

        # Get vector store and create index
        vector_store = get_weaviate_vector_store(client, class_name, text_key="text")
        if vector_store is None:
            return ToolResult(
                ok=False,
                tool_name="semantic_retrieve_tool",
                error=ToolError(
                    code="INITIALIZATION_ERROR",
                    message="Failed to initialize WeaviateVectorStore",
                ),
            )

        # Create VectorStoreIndex from vector store and query
        # Note: WeaviateVectorStore handles the actual querying
        index = VectorStoreIndex.from_vector_store(vector_store)
        query_bundle = QueryBundle(query_str=query)
        
        # Retrieve top_k results using retriever
        retriever = index.as_retriever(similarity_top_k=top_k)
        try:
            nodes_with_scores: List[NodeWithScore] = retriever.retrieve(query_bundle)
        except Exception as e:
            # If retrieval fails, return empty results
            return ToolResult(
                ok=True,
                tool_name="semantic_retrieve_tool",
                data={"matches": [], "query": query},
            )

        # Map results to expected format
        matches: List[Dict[str, Any]] = []
        for node_with_score in nodes_with_scores:
            node = node_with_score.node
            score = node_with_score.score or 0.0
            
            # Extract metadata
            metadata = node.metadata or {}
            chunk_id = metadata.get("chunk_id") or node.node_id
            text = node.get_content() or ""
            
            # Parse meta JSON if present
            meta_json = metadata.get("meta", "{}")
            try:
                meta = json.loads(meta_json) if isinstance(meta_json, str) else meta_json
            except (json.JSONDecodeError, TypeError):
                meta = metadata.copy()
            
            # Ensure meta has required fields
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
                    "score": float(score),
                    "meta": meta,
                }
            )

        # Sort by score (descending) - should already be sorted, but ensure
        matches.sort(key=lambda x: x["score"], reverse=True)

        return ToolResult(
            ok=True,
            tool_name="semantic_retrieve_tool",
            data={"matches": matches, "query": query},
        )

    except Exception as e:
        return ToolResult(
            ok=False,
            tool_name="semantic_retrieve_tool",
            error=ToolError(
                code="RUNTIME_ERROR",
                message=f"Failed to retrieve from Weaviate: {str(e)}",
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
