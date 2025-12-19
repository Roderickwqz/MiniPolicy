from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


EnvelopeType = Literal[
    "tool.request",
    "tool.result",
    "skill.result",
    "debate.turn",
    "verify.result",
    "score.result",
    "report.result",
    "evals.result",
]


class Envelope(BaseModel):
    """
    Phase-0 最小严格 Envelope：
    - 每个节点只允许追加 Envelope（不可就地改历史）
    - 所有 claim 必须携带 evidence_refs（chunk_id / graph_path_id）
    """

    envelope_type: EnvelopeType
    run_id: str
    node_id: str
    step_id: str  # 单调递增/或 uuid；Phase0 用简单计数也行
    ts_ms: int

    # 追溯链路（Evidence-first）
    graph_path_id: str  # e.g. "Input>IngestDocs>VectorIndex"
    chunk_ids: List[str] = Field(default_factory=list)

    # 核心载荷
    payload: Dict[str, Any] = Field(default_factory=dict)

    # 可选：错误信息
    error: Optional[Dict[str, Any]] = None
