from __future__ import annotations

import time
import uuid
from typing import Dict, Any, List

from app.core.envelope import Envelope
from app.core.state import RunState
from app.mcp.gateway import MCPGateway


def _now_ms() -> int:
    return int(time.time() * 1000)


def _step() -> str:
    return uuid.uuid4().hex[:12]


def _append_env(state: RunState, envs: List[Envelope]) -> None:
    state.setdefault("envelopes", [])
    state["envelopes"].extend([e.model_dump() for e in envs])


def node_input(state: RunState) -> RunState:
    # 输入节点只负责把 user_input 固化
    run_id = state["run_id"]
    graph_path_id = "Input"
    env = Envelope(
        envelope_type="skill.result",
        run_id=run_id,
        node_id="Input",
        step_id=_step(),
        ts_ms=_now_ms(),
        graph_path_id=graph_path_id,
        chunk_ids=[],
        payload={"accepted_input_keys": list(state.get("user_input", {}).keys())},
    )
    _append_env(state, [env])
    return state


def node_ingest_docs(state: RunState) -> RunState:
    """
    Phase0：把 input 里的 text 当作 1 个 chunk，生成 chunk_id
    """
    run_id = state["run_id"]
    graph_path_id = "Input>IngestDocs"
    text = (state.get("user_input") or {}).get("text", "")

    chunk_id = f"chunk_{uuid.uuid4().hex[:10]}"
    state.setdefault("chunks", {})
    state["chunks"][chunk_id] = {"text": text, "source": "user_input"}

    env = Envelope(
        envelope_type="skill.result",
        run_id=run_id,
        node_id="IngestDocs",
        step_id=_step(),
        ts_ms=_now_ms(),
        graph_path_id=graph_path_id,
        chunk_ids=[chunk_id],
        payload={"ingested": 1, "chunk_id": chunk_id},
    )
    _append_env(state, [env])
    return state


def node_vector_index(state: RunState) -> RunState:
    """
    Phase0：不做真实向量库，只做 placeholder（但保留 evidence 引用）
    """
    run_id = state["run_id"]
    graph_path_id = "Input>IngestDocs>VectorIndex"

    chunk_ids = list((state.get("chunks") or {}).keys())
    state["vector_index"] = {"type": "placeholder", "chunk_ids": chunk_ids}

    env = Envelope(
        envelope_type="skill.result",
        run_id=run_id,
        node_id="VectorIndex",
        step_id=_step(),
        ts_ms=_now_ms(),
        graph_path_id=graph_path_id,
        chunk_ids=chunk_ids,
        payload={"index_type": "placeholder", "indexed_chunks": len(chunk_ids)},
    )
    _append_env(state, [env])
    return state


def node_graph_build(state: RunState) -> RunState:
    run_id = state["run_id"]
    graph_path_id = "Input>IngestDocs>VectorIndex>GraphBuild"
    state["graph_build"] = {"type": "placeholder", "nodes": ["ParallelSkills", "Debate", "Verify"]}

    env = Envelope(
        envelope_type="skill.result",
        run_id=run_id,
        node_id="GraphBuild",
        step_id=_step(),
        ts_ms=_now_ms(),
        graph_path_id=graph_path_id,
        chunk_ids=list((state.get("chunks") or {}).keys()),
        payload={"graph_build": state["graph_build"]},
    )
    _append_env(state, [env])
    return state


def node_parallel_skills(state: RunState) -> RunState:
    """
    Phase0：假装跑了两个 skill，输出 claims（必须可回溯 chunk_id + graph_path_id）
    """
    run_id = state["run_id"]
    graph_path_id = "…>GraphBuild>ParallelSkills"
    chunk_ids = list((state.get("chunks") or {}).keys())

    claims = [
        {
            "claim": "Input 文本已被摄取并形成 chunk。",
            "evidence_refs": {"chunk_ids": chunk_ids, "graph_path_id": "Input>IngestDocs"},
        },
        {
            "claim": "当前 Phase0 使用 placeholder index（尚未接入真实向量库）。",
            "evidence_refs": {"chunk_ids": chunk_ids, "graph_path_id": "Input>IngestDocs>VectorIndex"},
        },
    ]

    env = Envelope(
        envelope_type="skill.result",
        run_id=run_id,
        node_id="ParallelSkills",
        step_id=_step(),
        ts_ms=_now_ms(),
        graph_path_id=graph_path_id,
        chunk_ids=chunk_ids,
        payload={"skills_ran": ["skill_a", "skill_b"], "claims": claims},
    )
    _append_env(state, [env])
    return state


def node_debate(state: RunState) -> RunState:
    run_id = state["run_id"]
    graph_path_id = "…>ParallelSkills>Debate"
    chunk_ids = list((state.get("chunks") or {}).keys())

    turn = Envelope(
        envelope_type="debate.turn",
        run_id=run_id,
        node_id="Debate",
        step_id=_step(),
        ts_ms=_now_ms(),
        graph_path_id=graph_path_id,
        chunk_ids=chunk_ids,
        payload={
            "bull": "骨架链路已满足 agent-first 的最小可运行验证。",
            "bear": "索引/检索/验证均为 placeholder，暂不具备真实 evidence 严谨性。",
        },
    )
    _append_env(state, [turn])
    return state


def node_verify(state: RunState) -> RunState:
    run_id = state["run_id"]
    graph_path_id = "…>Debate>Verify"
    chunk_ids = list((state.get("chunks") or {}).keys())

    # Phase0：验证规则=所有 claim 必须含 evidence_refs
    ok = True
    missing = []
    for e in state.get("envelopes", []):
        if e.get("envelope_type") == "skill.result":
            claims = (e.get("payload") or {}).get("claims") or []
            for c in claims:
                if "evidence_refs" not in c:
                    ok = False
                    missing.append(c)

    env = Envelope(
        envelope_type="verify.result",
        run_id=run_id,
        node_id="Verify",
        step_id=_step(),
        ts_ms=_now_ms(),
        graph_path_id=graph_path_id,
        chunk_ids=chunk_ids,
        payload={"ok": ok, "missing": missing},
    )
    _append_env(state, [env])
    return state


def node_score(state: RunState) -> RunState:
    run_id = state["run_id"]
    graph_path_id = "…>Verify>Score"
    score = 0.6  # Phase0：placeholder
    env = Envelope(
        envelope_type="score.result",
        run_id=run_id,
        node_id="Score",
        step_id=_step(),
        ts_ms=_now_ms(),
        graph_path_id=graph_path_id,
        chunk_ids=list((state.get("chunks") or {}).keys()),
        payload={"score": score, "rubric": {"phase0_skeleton": 0.6}},
    )
    _append_env(state, [env])
    return state


def node_report(state: RunState) -> RunState:
    """
    关键：写 report.md / artifacts.json 必须走 MCP gateway
    """
    gw = MCPGateway()
    run_id = state["run_id"]
    thread_id = state["thread_id"]
    graph_path_id = "…>Score>Report"
    step_id = _step()

    report_path = f"app/artifacts/{run_id}/report.md"
    artifacts_path = f"app/artifacts/{run_id}/artifacts.json"

    # 简单 report
    report_md = f"""# Phase0 Report

    - run_id: {run_id}
    - thread_id: {thread_id}

    ## Envelopes
    Total: {len(state.get("envelopes", []))}

    ## Notes
    This is Phase0 skeleton. VectorIndex/Verify/Score are placeholders.
    """

    req1, res1 = gw.call_tool(
        run_id=run_id,
        node_id="Report",
        step_id=step_id,
        graph_path_id=graph_path_id,
        tool_name="fs.write_text",
        tool_args={"path": report_path, "text": report_md},
        chunk_ids=list((state.get("chunks") or {}).keys()),
    )

    req2, res2 = gw.call_tool(
        run_id=run_id,
        node_id="Report",
        step_id=step_id,
        graph_path_id=graph_path_id,
        tool_name="fs.write_json",
        tool_args={"path": artifacts_path, "data": {"envelopes": state.get("envelopes", [])}},
        chunk_ids=list((state.get("chunks") or {}).keys()),
    )

    _append_env(state, [req1, res1, req2, res2])

    state["report_path"] = report_path
    state["artifacts_path"] = artifacts_path

    env = Envelope(
        envelope_type="report.result",
        run_id=run_id,
        node_id="Report",
        step_id=_step(),
        ts_ms=_now_ms(),
        graph_path_id=graph_path_id,
        chunk_ids=list((state.get("chunks") or {}).keys()),
        payload={"report_path": report_path, "artifacts_path": artifacts_path},
    )
    _append_env(state, [env])
    return state


def node_evals(state: RunState) -> RunState:
    run_id = state["run_id"]
    graph_path_id = "…>Report>Evals"
    env = Envelope(
        envelope_type="evals.result",
        run_id=run_id,
        node_id="Evals",
        step_id=_step(),
        ts_ms=_now_ms(),
        graph_path_id=graph_path_id,
        chunk_ids=list((state.get("chunks") or {}).keys()),
        payload={"phase0": {"pass": True, "checks": ["graph_ran", "artifacts_written", "envelopes_present"]}},
    )
    _append_env(state, [env])
    return state
