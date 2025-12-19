# PRD v1.1 — Mini-Policy / Law IC Lab (Full-stack Compliance Due Diligence Demo)

## 1. Overview

**One-liner:** Input **(Product/Business Brief + Jurisdiction(s) + Regulation PDFs)** → system executes **RAG + GraphRAG + multi-skill analysis + IC Debate + Verification + Scoring + Async + SSE Web UI + Export (MD/PDF) + Evals + Memory + DSPy optimization**.

**Not legal advice.** This is an engineering demo emphasizing traceable evidence and measurable quality.

## 2. Goals

- Demonstrate a complete Agent/Workflow/Infra pipeline with minimal offline data (5–8 PDFs).
- Ensure **evidence-bound, verifiable** outputs (chunk IDs / graph path IDs).
- Provide **real-time execution visibility** and exportable deliverables (MD/PDF).
- Provide evaluation + self-improvement loop (Evals + DSPy).
- (Architect alignment) Support **resumability + multi-run concurrency**, plus deep observability (traces/logs/graph viz).

## 3. Non-Goals

- Acting as a licensed legal advisor or guaranteeing legal correctness.
- Unrestricted internet crawling / real-time regulatory monitoring.
- Full jurisdictional completeness beyond the provided corpus (unless explicitly added to doc registry).

## 4. Users & Use Cases

### Personas
- **Builder / Demo Owner:** wants an impressive end-to-end showcase.
- **IC Reviewer:** wants debate, verified evidence, and go/hold/avoid grade.
- **Engineer Reviewer:** wants architecture, guardrails, evals, observability.

### Top Use Cases
1. Create a run from product brief + jurisdictions + selected PDFs.
2. Watch real-time pipeline progress and debate timeline.
3. Inspect evidence (chunks/graph paths) for every claim.
4. Export report as **Markdown + PDF**.
5. Compare runs, reuse memory, and observe eval metrics trend over time.

---

## 5. Solution Overview (Architect-aligned)

### 5.1 High-level Topology

The architecture includes: **Streamlit UI**, **FastAPI**, **SSE + WebSockets**, **Tool Permission + Guardrails Gateway**, **LangGraph runtime + checkpointer**, **Celery + Redis**, **Vector DB (Chroma/FAISS)**, **Neo4j + GraphQL**, and **Tracing/Evals**.

`flowchart LR   UI[Streamlit UI + Trace/Graph Viewers] --> API[FastAPI]   API --> SSE[SSE /runs/{id}/events]   API --> WS[WebSockets (HITL approvals)]   API --> AUTH[Tool Permission + Guardrails Gateway]   API --> Q[Celery Queue]   Q --> R[Redis]   Q --> LG[LangGraph Runtime]   LG --> CP[Checkpointer]   LG --> MCP[MCP Tools]   MCP --> VDB[Chroma/FAISS]   MCP --> N4J[Neo4j GraphRAG]   N4J --> GQL[GraphQL for Graph UI]   LG --> OBS[Tracing/Evals]`

### 5.2 Orchestration “Supervisor” Concepts
- **ToolFlow:** tool routing rules, permission checks, retries, fallback.
- **Workflow Models:** choose orchestration templates by run type (due diligence / mapping / comparison).

### 5.3 Evidence & Traceability Contract
- Every user-visible claim must link to **chunk_id** and/or **graph_path_id** for replay + UI drill-down.

---

# 6. FR — Functional Requirements (BMAD)
> **Legend:** Priority = P0 (Must) / P1 (Should) / P2 (Could)

### FR-001 Run creation (P0)
- Users can create a run with `product_brief`, `jurisdictions[]`, `doc_registry[]`.
- Acceptance: `POST /runs` returns `run_id`; run enters `RUN_STARTED`.

### FR-002 Async execution (Celery) (P0)
- Runs execute asynchronously in worker(s).
- Acceptance: API returns immediately; worker processes LangGraph; failures retried and surfaced.

### FR-003 Real-time events via SSE (P0)
- Provide `GET /runs/{id}/events` (SSE).
- Acceptance: ordered events: `NODE_STARTED/FINISHED`, `SKILL_OUTPUT`, `SCORE_UPDATED`, `REPORT_READY`.

### FR-003b WebSockets for HITL + interactive UX (P1)
- Provide WebSockets for interactive approvals + live exploration (graph browsing, chat-like sessions).
- Acceptance: UI can submit/receive “approval decisions” for flagged nodes; run can resume from checkpoint.

### FR-004 LangGraph state machine + checkpointing (P0)
- Orchestrate pipeline using LangGraph with persisted checkpoints; run can resume after interruption.

### FR-004b Human-in-the-loop edges (P1)
- Approval edge after verification failures / red-flag triggers; reviewer can override uncertain claims.
- Acceptance: run pauses in a “REVIEW_REQUIRED” state until WS approval or timeout policy.

### FR-005 Tool Permission + Guardrails Gateway (P0)
- All tool calls must pass through a permission + guardrails gateway (per-tenant/per-run tool access control).
- Acceptance: unauthorized tool calls are blocked and logged; run marks tool failure clearly.

### FR-006 MCP unified tool layer (no direct IO) (P0)
- All IO goes through MCP tools; nodes/skills cannot directly read/write external resources.

### FR-007 PDF ingest (MCP tool) (P0)
- `pdf_ingest_tool` chunks PDFs and registers doc metadata; produces stable `chunk_id`.

### FR-008 RAG vector indexing + retrieve skill (P0)
- Build vector index from chunks; retrieval returns top-k with scores + chunk metadata.
- `SemanticRetrieve` returns strict JSON; evidence contains `chunk_id`.

### FR-009 GraphRAG build + provenance (P0)
- Build minimal Neo4j graph; derived nodes link back to one/many Chunk nodes (provenance).

### FR-010 GraphRAG querying + path IDs (P0)
- Graph queries return `graph_path_id` (stored for replay and UI visualization).

### FR-010b Graph Visualization API (GraphQL) (P1)
- Provide GraphQL endpoints over Neo4j to support graph visualization UI.
- Acceptance: UI can query nodes/edges + provenance, drill down by `graph_path_id`.

### FR-011 Parallel analysis skills (P0)
- Run applicability/obligations/risks/penalties in parallel; merge into `findings`.

### FR-012 Mitigations (P0)
- Mitigations include assumptions/unknowns and evidence binding where possible.

### FR-013 IC Debate (Bull/Bear/Moderator) (P0)
- Debate produces bull points, bear points, disputed points, decision, confidence; moderator forces evidence binding.

### FR-014 Verification (Evidence-Enforcer) (P0)
- Re-check claims using RAG re-retrieval + GraphRAG path validation; label `verified/uncertain/failed`.

### FR-015 Scoring (P0)
- Multi-dim subscores + confidence; confidence decreases with verification failures / low evidence strength.

### FR-016 Guardrails (Input/Retrieval/Output) (P0)
- Input guard, retrieval source validation, output redaction + uncertainty labeling.
- Architect adds: sandbox execution + red-flag detector.

### FR-017 Report export (Markdown + PDF) (P0)
- Generate `report.md`, `report.pdf`, `artifacts.json`; report includes verification table + evidence refs.

### FR-018 Web UI (P1)
- UI shows progress, debate timeline, scoring, evidence explorer, graph visualization.
- Architect adds viewers: ReAct trace viewer, LangGraph Studio, agent logs viewer.

### FR-019 Evals (P0)
- Store metrics per run: schema pass rate, evidence coverage, verification fail rate, cost/latency; optional cross-model consistency.

### FR-020 Memory + cache (P1)

- Maintain semantic/episodic/graph memory + caching to improve latency on reruns.

### FR-020b Caching policy detail (P1)
- Add explicit caches: semantic cache, analysis cache, context/KV cache; web caching TTL 1–7 days when web fetch enabled.

### FR-021 DSPy optimization (P1)
- Optimize ≥2 modules (AssessApplicability, ExtractObligations); at least one eval metric improves.

---

# 7. NFR — Non-Functional Requirements (Architect-aligned)

### NFR-001 Offline-first (default) + optional controlled web sources (P0)
- Default mode runs offline on local PDFs; optional web fetch is **sandboxed** and gated by permissions/guardrails.
### NFR-002 Reliability & resumability (P0)
- Retries for MCP tool calls; LangGraph checkpoints allow resuming.
### NFR-003 Observability & auditability (P0)
- Structured logs + `audit_log`; trace integration and run replay support.
### NFR-004 Security (P0)
- File-type checks, input sanitization, least-privilege credentials for Neo4j/DB.
- Tool permission + sandbox execution + red-flag detector.
### NFR-005 Performance (demo target) (P0)
- With 5–8 PDFs and caching, run completes within demo-acceptable time and streams progress continuously.
### NFR-006 Multi-run concurrency (P1)
- Support concurrent runs and isolation (per tenant/project); fair scheduling/quotas as needed.    

---

# 8. Data Requirements (Offline Corpus)

- 5–8 reusable PDFs (e.g., GDPR, EU AI Act, CCPA/CPRA, regulator guidance/FAQ, enforcement summaries).    

---

# 9. Data Contracts

### 9.1 Skill Output Envelope (JSON Mode) (P0)
All skills must output a strict envelope containing:
- `items[]`
- `confidence`
- `evidence[]` (chunk_id and/or graph_path_id)
- `assumptions[]`
- `unknowns[]`
### 9.2 Graph Schema (minimal) (P0)
- Nodes: `Doc`, `Chunk`, `Law`, `Article`, `Obligation`, `Risk`, `Penalty`, `Concept`, `Product`, `Company`
- Edges: `HAS_CHUNK`, `MENTIONS`, `REQUIRES`, `TRIGGERS`, `SUBJECT_TO`, `LEADS_TO`, `MITIGATES`, `SUPPORTS`
### 9.3 Event Schema (SSE/WS) (P1)
- SSE events at minimum: `NODE_STARTED`, `NODE_FINISHED`, `SKILL_OUTPUT`, `SCORE_UPDATED`, `REPORT_READY`.
- WS events additionally: `APPROVAL_REQUESTED`, `APPROVAL_GRANTED`, `APPROVAL_DENIED`, `RUN_RESUMED`    

---
# 10. API

Minimum:
- `POST /runs`
- `GET /runs/{id}`
- `GET /runs/{id}/events` (SSE)
- `GET /runs/{id}/report.pdf`

Additions (Architect-aligned):
- `WS /runs/{id}/ws` for approvals + interactive UX.
- `/graphql` (or similar) for Neo4j graph visualization queries.

---

# 11. Delivery Plan

Keep the existing 7-day plan, with explicit scope additions:
- Day 6 includes **SSE + WS (HITL)** and basic trace/log viewers.
- Day 7 includes **cache policy + guardrails integration** details.