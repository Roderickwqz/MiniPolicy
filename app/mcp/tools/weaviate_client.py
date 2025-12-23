# app/mcp/tools/weaviate_client.py
from __future__ import annotations

from typing import Optional, Sequence

import weaviate
from weaviate.classes.config import Property, DataType, Configure
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from app.mcp.config import get_weaviate_client_config


def get_weaviate_client() -> weaviate.WeaviateClient:
    """
    Get a Weaviate v4 client instance (v4-only).
    Raises exception if connection fails.
    """
    config = get_weaviate_client_config()
    print("weaviate_client_config =", config)
    return weaviate.connect_to_local(
        host=config["host"],
        port=config["port"],              # REST
        grpc_port=config["grpc_port"],    
    )


def ensure_weaviate_collection(
    client: weaviate.WeaviateClient,
    class_name: str,
    *,
    text_key: str = "text",
    properties: Sequence[Property] | None = None,
    vectorizer_mode: str = "none",  # "none" or "text2vec-openai"
    embedding_model: str = "text-embedding-3-small",
) -> None:
    if properties is None:
        properties = []

    # exists?
    try:
        client.collections.get(class_name)
        return
    except Exception:
        pass

    props = list(properties)
    if not any(p.name == text_key for p in props):
        props.insert(0, Property(name=text_key, data_type=DataType.TEXT))

    if vectorizer_mode == "none":
        vectorizer_config = Configure.Vectorizer.none()
    elif vectorizer_mode in ("text2vec-openai", "openai"):
        vectorizer_config = Configure.Vectorizer.text2vec_openai(model=embedding_model)
    else:
        raise ValueError(f"Unknown vectorizer_mode: {vectorizer_mode}")

    client.collections.create(
        name=class_name,
        properties=props,
        vectorizer_config=vectorizer_config,
    )


def get_weaviate_vector_store(
    client: weaviate.WeaviateClient,
    class_name: str,
    text_key: str = "text",
) -> Optional[WeaviateVectorStore]:
    """
    Get a LlamaIndex WeaviateVectorStore instance (adapter only).
    """
    return WeaviateVectorStore(
        weaviate_client=client,
        index_name=class_name,
        text_key=text_key,
    )
