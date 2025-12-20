from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class RunState(TypedDict, total=False):
    # Identity / routing
    run_id: str
    thread_id: str

    # Input
    user_input: Dict[str, Any]

    # Evidence objects（Phase0 用最简）
    chunks: Dict[str, Dict[str, Any]]      # chunk_id -> {text, meta}
    ingest_summary: Dict[str, Any]
    vector_index: Dict[str, Any]           # {"index_name": ..., "chunk_ids": [...]}
    retrieval: Dict[str, Any]
    graph_build: Dict[str, Any]            # placeholder / graph meta

    # Outputs（严格只追加）
    envelopes: List[Dict[str, Any]]        # 存 Envelope.dict()

    # Artifacts paths
    report_path: Optional[str]
    artifacts_path: Optional[str]
