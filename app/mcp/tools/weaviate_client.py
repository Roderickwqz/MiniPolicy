# app/mcp/tools/weaviate_client.py
from __future__ import annotations

import os
from typing import Optional
import weaviate
from llama_index.vector_stores.weaviate import WeaviateVectorStore

# Weaviate connection configuration
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "22006"))
WEAVIATE_SCHEME = os.getenv("WEAVIATE_SCHEME", "http")
WEAVIATE_URL = f"{WEAVIATE_SCHEME}://{WEAVIATE_HOST}:{WEAVIATE_PORT}"


def get_weaviate_client():
    """
    Get a Weaviate client instance.
    Returns None if weaviate is not installed.
    """
    try:
        # Try v4 API first (newer versions)
        # return weaviate.Client(
        #     url=WEAVIATE_URL,
        #     auth_client_secret=None,  # No auth (AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: true)
        # )
        return weaviate.connect_to_local(
            host="localhost",
            port=22006,      # HTTP/REST
            grpc_port=22007 # gRPC
        )
    except Exception:
        try:
            # Fallback to v3 API
            return weaviate.Client(WEAVIATE_URL)
        except Exception:
            return None


def get_weaviate_vector_store(client: weaviate.Client, class_name: str, text_key: str = "text") -> Optional[WeaviateVectorStore]:
    """
    Get a LlamaIndex WeaviateVectorStore instance.
    
    Args:
        class_name: The Weaviate class name to use
        text_key: The property name for text content (default: "text")
    
    Returns:
        WeaviateVectorStore instance or None if not available
    """
    if WeaviateVectorStore is None:
        return None
    
    # client = get_weaviate_client()
    # if client is None:
    #     return None
    
    return WeaviateVectorStore(
        weaviate_client=client,
        index_name=class_name,
        text_key=text_key,
    )


def ensure_weaviate_class(client, class_name: str, properties: list[dict]):
    """
    Ensure a Weaviate class exists with the specified properties.
    Creates the class if it doesn't exist.
    
    Args:
        client: Weaviate client instance
        class_name: Name of the class to create/verify
        properties: List of property definitions
    """
    if client is None:
        return
    
    # Check if class exists
    try:
        if hasattr(client, "collections"):
            # v4 API
            try:
                collection = client.collections.get(class_name)
                # If we can get it, it exists
                return
            except Exception:
                # Collection doesn't exist, will create it
                pass
        else:
            # v3 API
            schema = client.schema.get()
            existing_classes = [cls.get("class") for cls in schema.get("classes", [])]
            if class_name in existing_classes:
                return
    except Exception:
        # If we can't check, try to create anyway
        pass
    
    # Create class with properties
    try:
        if hasattr(client, "collections"):
            # v4 API - WeaviateVectorStore will handle collection creation
            # We just need to ensure it doesn't exist check fails
            # The actual creation will be done by LlamaIndex's WeaviateVectorStore
            pass
        else:
            # v3 API - create class schema
            class_schema = {
                "class": class_name,
                "description": f"Vector store for {class_name}",
                "vectorizer": "text2vec-openai",  # From docker-compose config
                "properties": properties,
            }
            client.schema.create_class(class_schema)
    except Exception as e:
        # Class might have been created by another process, or error creating
        # We'll let the vector store handle it
        pass
