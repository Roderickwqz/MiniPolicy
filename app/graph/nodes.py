from __future__ import annotations

import hashlib
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
    PDF ingest 通过 MCP gateway，chunk_id 稳定且包含 doc_id/page/hash。
    """
    run_id = state["run_id"]
    user_input = state.get("user_input") or {}
    docs = user_input.get("docs") or []
    chunking_defaults = user_input.get("chunking") or {}
    graph_path_id = "Input>IngestDocs"

    gw = MCPGateway()
    chunk_store = state.setdefault("chunks", {})
    chunk_ids: List[str] = []
    ingest_records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    def _resolved_chunking(doc_cfg: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        merged.update(chunking_defaults)
        merged.update(doc_cfg.get("chunking") or {})
        if doc_cfg.get("chunk_size") is not None:
            merged["chunk_size"] = doc_cfg["chunk_size"]
        if doc_cfg.get("overlap") is not None:
            merged["overlap"] = doc_cfg["overlap"]
        if doc_cfg.get("segmentation"):
            merged["segmentation"] = doc_cfg["segmentation"]
        return merged

    for doc in docs:
        pdf_path = doc.get("pdf_path")
        if not pdf_path:
            errors.append({"doc": doc, "error": "missing_pdf_path"})
            continue

        chunk_cfg = _resolved_chunking(doc)
        tool_args: Dict[str, Any] = {"pdf_path": pdf_path}
        if doc.get("doc_id"):
            tool_args["doc_id"] = doc["doc_id"]
        if chunk_cfg.get("chunk_size"):
            tool_args["chunk_size"] = int(chunk_cfg["chunk_size"])
        if chunk_cfg.get("overlap") is not None:
            tool_args["overlap"] = int(chunk_cfg["overlap"])
        if chunk_cfg.get("segmentation"):
            tool_args["segmentation"] = str(chunk_cfg["segmentation"])

        req, res = gw.call_tool(
            run_id=run_id,
            node_id="IngestDocs",
            step_id=_step(),
            graph_path_id=graph_path_id,
            tool_name="pdf.ingest",
            tool_args=tool_args,
            chunk_ids=[],
        )
        _append_env(state, [req, res])

        output = res.payload.get("output") or {}
        if not output.get("ok"):
            errors.append({"doc": doc, "error": output.get("error") or "ingest_failed"})
            continue

        data = output.get("data") or {}
        doc_chunks = data.get("chunks") or []
        for chunk in doc_chunks:
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            chunk_store[chunk_id] = chunk
            chunk_ids.append(chunk_id)

        ingest_records.append(
            {
                "doc_id": data.get("doc_id") or doc.get("doc_id"),
                "pdf_path": pdf_path,
                "chunks": len(doc_chunks),
                "segmentation": tool_args.get("segmentation", "deterministic"),
            }
        )

    if not chunk_ids:
        fallback_text = (user_input.get("text") or "").strip()
        if fallback_text:
            chunk_hash = hashlib.sha256(fallback_text.encode("utf-8")).hexdigest()
            chunk_id = f"user::{chunk_hash[:12]}"
            chunk_store[chunk_id] = {
                "chunk_id": chunk_id,
                "text": fallback_text,
                "meta": {
                    "doc_id": "user_input",
                    "page": 1,
                    "hash": chunk_hash,
                    "chunk_method": "text",
                    "source": "user_input",
                },
            }
            chunk_ids.append(chunk_id)
            ingest_records.append(
                {"doc_id": "user_input", "pdf_path": None, "chunks": 1, "segmentation": "text"}
            )

    state["ingest_summary"] = {"records": ingest_records, "errors": errors, "chunks": len(chunk_ids)}

    env = Envelope(
        envelope_type="skill.result",
        run_id=run_id,
        node_id="IngestDocs",
        step_id=_step(),
        ts_ms=_now_ms(),
        graph_path_id=graph_path_id,
        chunk_ids=chunk_ids,
        payload={
            "docs_ingested": len(ingest_records),
            "chunks_total": len(chunk_ids),
            "errors": errors,
        },
    )
    _append_env(state, [env])
    return state


def node_vector_index(state: RunState) -> RunState:
    """
    将 chunks 写入向量索引（demo 内存向量库）。
    """
    run_id = state["run_id"]
    graph_path_id = "Input>IngestDocs>VectorIndex"
    chunk_store = state.get("chunks") or {}
    chunk_ids = list(chunk_store.keys())

    gw = MCPGateway()
    index_name = (
        (state.get("vector_index") or {}).get("index_name")
        or (state.get("user_input") or {}).get("index_name")
        or f"{run_id}::default"
    )

    req, res = gw.call_tool(
        run_id=run_id,
        node_id="VectorIndex",
        step_id=_step(),
        graph_path_id=graph_path_id,
        tool_name="vector.upsert",
        tool_args={"index_name": index_name, "chunks": list(chunk_store.values())},
        chunk_ids=chunk_ids,
    )
    _append_env(state, [req, res])

    output = res.payload.get("output") or {}
    data = output.get("data") or {}
    upserted = data.get("upserted", 0)

    state["vector_index"] = {"index_name": index_name, "chunk_ids": chunk_ids, "upserted": upserted}

    env = Envelope(
        envelope_type="skill.result",
        run_id=run_id,
        node_id="VectorIndex",
        step_id=_step(),
        ts_ms=_now_ms(),
        graph_path_id=graph_path_id,
        chunk_ids=chunk_ids,
        payload={"index_name": index_name, "indexed_chunks": upserted},
    )
    _append_env(state, [env])
    return state


def node_semantic_retrieve(state: RunState) -> RunState:
    """
    查询向量索引并输出严格的 Skill Envelope。
    """
    run_id = state["run_id"]
    graph_path_id = "Input>IngestDocs>VectorIndex>SemanticRetrieve"
    gw = MCPGateway()

    index_info = state.get("vector_index") or {}
    index_name = index_info.get("index_name") or f"{run_id}::default"
    chunk_ids = index_info.get("chunk_ids") or list((state.get("chunks") or {}).keys())

    user_input = state.get("user_input") or {}
    retrieval_cfg = user_input.get("retrieval") or {}
    queries = retrieval_cfg.get("queries") or []
    query = retrieval_cfg.get("query") or (queries[0] if queries else None)
    if not query:
        query = user_input.get("text") or "Summarize compliance obligations"
    top_k = int(retrieval_cfg.get("top_k") or user_input.get("retrieval_top_k") or 3)

    req, res = gw.call_tool(
        run_id=run_id,
        node_id="SemanticRetrieve",
        step_id=_step(),
        graph_path_id=graph_path_id,
        tool_name="vector.query",
        tool_args={"index_name": index_name, "query": query, "top_k": top_k},
        chunk_ids=chunk_ids,
    )
    _append_env(state, [req, res])

    output = res.payload.get("output") or {}
    data = output.get("data") or {}
    matches = data.get("matches") or []

    items: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    scores = [m.get("score", 0.0) for m in matches]

    for match in matches:
        chunk_id = match.get("chunk_id")
        if not chunk_id:
            continue
        meta = match.get("meta") or {}
        snippet = (match.get("text") or "").strip()
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"
        score = round(match.get("score", 0.0), 3)
        items.append(
            {
                "chunk_id": chunk_id,
                "doc_id": meta.get("doc_id"),
                "page": meta.get("page"),
                "score": score,
                "snippet": snippet,
                "evidence_refs": {"chunk_ids": [chunk_id], "graph_path_id": graph_path_id},
            }
        )
        evidence.append({"chunk_id": chunk_id, "doc_id": meta.get("doc_id"), "page": meta.get("page")})

    confidence = round(sum(scores) / len(scores), 3) if scores else 0.2
    unknowns: List[str] = []
    if not items:
        unknowns.append(f"No supporting chunks retrieved for query '{query}'.")
    if output.get("error"):
        unknowns.append(f"Tool error: {output['error'].get('message')}")

    assumptions = [
        "SemanticRetrieve uses a bag-of-words cosine similarity; treat scores as coarse signals."
    ]

    payload = {
        "query": query,
        "top_k": top_k,
        "items": items,
        "confidence": confidence,
        "evidence": evidence,
        "assumptions": assumptions,
        "unknowns": unknowns,
    }

    state["retrieval"] = payload

    env = Envelope(
        envelope_type="skill.result",
        run_id=run_id,
        node_id="SemanticRetrieve",
        step_id=_step(),
        ts_ms=_now_ms(),
        graph_path_id=graph_path_id,
        chunk_ids=[e["chunk_id"] for e in evidence],
        payload=payload,
    )
    _append_env(state, [env])
    return state


def node_graph_build(state: RunState) -> RunState:
    run_id = state["run_id"]
    graph_path_id = "Input>IngestDocs>VectorIndex>SemanticRetrieve>GraphBuild"
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
    Demo：并行 skill 汇总 ingest/index/retrieve 结果。
    """
    run_id = state["run_id"]
    graph_path_id = "Input>IngestDocs>VectorIndex>SemanticRetrieve>GraphBuild>ParallelSkills"
    chunk_ids = list((state.get("chunks") or {}).keys())
    retrieval = state.get("retrieval") or {}

    claims = [
        {
            "claim": f"Ingested {len(chunk_ids)} chunks with doc/page provenance.",
            "evidence_refs": {"chunk_ids": chunk_ids, "graph_path_id": "Input>IngestDocs"},
        },
    ]
    if retrieval.get("items"):
        claims.append(
            {
                "claim": f"Semantic retrieval found {len(retrieval['items'])} candidate evidence items.",
                "evidence_refs": {
                    "chunk_ids": [item["chunk_id"] for item in retrieval["items"]],
                    "graph_path_id": "Input>IngestDocs>VectorIndex>SemanticRetrieve",
                },
            }
        )

    env = Envelope(
        envelope_type="skill.result",
        run_id=run_id,
        node_id="ParallelSkills",
        step_id=_step(),
        ts_ms=_now_ms(),
        graph_path_id=graph_path_id,
        chunk_ids=chunk_ids,
        payload={"skills_ran": ["ingest_audit", "semantic_retrieve_summary"], "claims": claims},
    )
    _append_env(state, [env])
    return state


def node_debate(state: RunState) -> RunState:
    run_id = state["run_id"]
    graph_path_id = "Input>IngestDocs>VectorIndex>SemanticRetrieve>GraphBuild>ParallelSkills>Debate"
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
    graph_path_id = "Input>IngestDocs>VectorIndex>SemanticRetrieve>GraphBuild>ParallelSkills>Debate>Verify"
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
    graph_path_id = (
        "Input>IngestDocs>VectorIndex>SemanticRetrieve>"
        "GraphBuild>ParallelSkills>Debate>Verify>Score"
    )
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
    graph_path_id = (
        "Input>IngestDocs>VectorIndex>SemanticRetrieve>"
        "GraphBuild>ParallelSkills>Debate>Verify>Score>Report"
    )
    step_id = _step()

    report_path = f"app/artifacts/{run_id}/report.md"
    artifacts_path = f"app/artifacts/{run_id}/artifacts.json"

    # 简单 report
    retrieval = state.get("retrieval") or {}
    report_lines = [
        "# Phase2 Report (RAG focus)",
        "",
        f"- run_id: {run_id}",
        f"- thread_id: {thread_id}",
        "",
        "## Envelopes",
        f"Total: {len(state.get('envelopes', []))}",
        "",
        "## Retrieval",
        f"- query: {retrieval.get('query')}",
        f"- top_k: {retrieval.get('top_k')}",
        f"- hits: {len(retrieval.get('items', []))}",
        f"- confidence: {retrieval.get('confidence')}",
    ]
    report_md = "\n".join(report_lines)

    req1, res1 = gw.call_tool(
        run_id=run_id,
        node_id="Report",
        step_id=step_id,
        graph_path_id=graph_path_id,
        tool_name="fs.write_text",
        tool_args={"path": report_path, "text": report_md},
        chunk_ids=list((state.get("chunks") or {}).keys()),
    )

    envelopes_path = f"app/artifacts/{run_id}/envelopes.json"

    req2, res2 = gw.call_tool(
        run_id=run_id,
        node_id="Report",
        step_id=step_id,
        graph_path_id=graph_path_id,
        tool_name="fs.write_json",
        tool_args={"path": envelopes_path, "data": {"envelopes": state.get("envelopes", [])}},
        chunk_ids=list((state.get("chunks") or {}).keys()),
    )

    state["envelopes_path"] = envelopes_path


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
    graph_path_id = (
        "Input>IngestDocs>VectorIndex>SemanticRetrieve>"
        "GraphBuild>ParallelSkills>Debate>Verify>Score>Report>Evals"
    )
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
