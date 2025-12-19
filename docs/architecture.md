# Architecture — Agentic Compliance Due Diligence System (RAG + GraphRAG + Debate + Verification + Evals)

## 0. Executive Summary

This system performs **compliance due diligence** for a product/company using **offline documents + optional controlled web sources**, producing **evidence-bound** outputs with **RAG + GraphRAG**, **IC-style Debate**, **Verification**, **Scoring**, and **Markdown/PDF exports**. It supports **async execution** with real-time UI updates via **SSE/WebSockets**, plus full **tracing, monitoring, evals, memory, caching, and guardrails**.

---

## 1. Goals, Non-Goals, and Constraints

### Goals
- Build a full **agent engineering** demo: ingestion → retrieval → analysis → debate → verification → scoring → reporting.
- Every output claim must be tied to **evidence IDs** (chunk IDs and/or graph path IDs).
- Provide **real-time observability**: agent logs, traces, graph visualization, and ReAct timeline.
- Support **multi-tenant / multi-project concurrency** and resumability.

### Non-Goals (default mode)
- No unrestricted internet crawling by default; online tools are **optional and sandboxed**.

---

## 2. System Topology (High-Level)

`flowchart LR   subgraph UI["Web UI Layer"]     ST[Streamlit UI]     TR[ReAct trace viewer]     LGV[LangGraph Studio]     LOGV[Agent logs viewer]     GRV[GraphRAG Visualization UI]   end    subgraph API["Backend/API Layer"]     FA[FastAPI]     SSE[SSE Endpoint]     WS[WebSockets Endpoint]     AUTH[Tool Permission + Guardrails Gateway]   end    subgraph ORCH["Orchestration Layer"]     LG[LangGraph Runtime]     CP[LangGraph checkpointer]     TF[ToolFlow (Supervisor Tool Flow)]     WM[Workflow Models (Supervisor Workflow Models)]   end    subgraph ASYNC["Async/Queue Layer"]     CE[Celery]     RD[Redis]   end    subgraph DATA["Data/Retrieval Layer"]     VX[Vector DB: Chroma / FAISS]     N4J[Neo4j (Graph Memory + GraphRAG)]     GQL[neo4j-graphql (GraphQL over Neo4j)]   end    subgraph TOOLS["MCP Tools Layer"]     MCPc[MCP Client]     MCPs[MCP Server]     MCPstr[MCP streaming]     PB[parallel batching]     WGET[requests/httpx Web Fetch (DataAgent)]   end    subgraph OBS["Tracing / Monitoring / Evals"]     LS[LangSmith]     PX[Arize Phoenix]     HL[Helicone]     EV[Self-hosted eval scripts]   end    ST --> FA   FA --> SSE   FA --> WS   FA --> CE   CE --> RD   CE --> LG   LG --> CP   LG --> TF   LG --> WM   LG --> MCPc   MCPc --> MCPs   MCPs --> VX   MCPs --> N4J   MCPs --> WGET   N4J --> GQL   LG --> LS   LG --> PX   LG --> HL   LG --> EV    TR --> LS   LGV --> LG   LOGV --> FA   GRV --> N4J`

---

## 3. Required Tools (Agent Engineering)

### 3.1 Supervisor Tools (Stage 3)
- **ToolFlow (Stage 3: Supervisor中的工具流)**
    - Defines tool routing rules, permission checks, retries, and fallback strategies.
- **Workflow Models (Stage 3: Supervisor中的工作流模型)**
    - Selects orchestration templates based on task type (due diligence / policy mapping / comparison runs).
### 3.2 Visualization & Debugging Tools (Stage 12)
- **ReAct trace viewer (Stage 12)**
    - Visualizes step-by-step ReAct reasoning + tool calls + evidence references.
- **LangGraph Studio (Stage 12)**
    - Visualizes nodes/edges/state transitions; supports replay/resume debugging.
- **Agent logs viewer (Stage 12)**
    - Shows per-run structured logs, tool usage, errors, latency, cost.
- **GraphRAG Visualization UI (Stage 12)**
    - Graph browsing: nodes/edges, provenance, graph_path_id drill-down.

---

## 4. Libraries / Frameworks

### 4.1 Core Language & Runtime
- **Python (Stage 1)**
- **asyncio (Stage 5)** for concurrent DataAgent operations.

### 4.2 Web Fetching
- **requests (Stage 2)** — basic web fetching.
- **httpx (Stage 2 / Stage 5)** — async fetching and used by DataAgent.

> Note: Web fetching is **optional** and governed by guardrails + caching + sandbox execution.

### 4.3 RAG / Agent Frameworks
- **LlamaIndex (Stage 2)** — RAG pipelines, indexing, retrieval utilities.
- **LangChain (Stage 2)** — tool abstractions, prompt routing, RAG building blocks.
- **LangGraph (Stage 3/4/7)** — state machine, nodes/edges, parallel nodes, multi-turn conversation nodes.

### 4.4 Vector Stores
- **Chroma (Stage 2 / Stage 5)** — vector DB for document store and memory index.
- **FAISS (Stage 2 / Stage 5)** — vector DB alternative for local deployments.

### 4.5 Graph Layer
- **Neo4j (Stage 6 / Stage 11)** — knowledge graph + graph memory.
- **neo4j-graphql (Stage 6)** — GraphQL queries over Neo4j for UI and API usage.

### 4.6 Async Tasks & Backend/UI
- **Celery (Stage 12)** — async task queue.
- **Redis (Stage 12)** — Celery broker/backend and event buffering.
- **FastAPI (Stage 12)** — backend API (runs, events, downloads).
- **Streamlit (Stage 12)** — Web UI.

---

## 5. Models / LLM Providers (Stage 1 & Stage 4)

### Providers
- **OpenAI (Stage 1: LLM调用)**
- **Other LLM (Stage 1: 通用LLM)** — pluggable provider interface.
### Model Roles
- **o3 / o3-mini (Stage 4)** — strong reasoning for **Debate / IC** and verification-heavy steps.
- **GPT-4.1 (Stage 4)** — summarization, drafting, structure normalization.
- **Claude Sonnet (Stage 4)** — alternative summarization + cross-model consistency checks.

> Architecture supports multi-model routing policies per node/skill.

---

## 6. Databases
- **Vector DB (Stage 2)** — Chroma/FAISS for retrieval and semantic memory.
- **Neo4j (Stage 6 / Stage 11)** — GraphRAG storage + Graph Memory.
- **LangGraph checkpointer (Stage 3)** — persistent state checkpointing for resume/replay.

---

## 7. APIs and Services (Agent Engineering)
- **Server-Sent Events (SSE) (Stage 12)**
    - Streaming progress updates: node start/finish, skill output, score updates, report ready.
- **WebSockets (Stage 12)**
    - Bidirectional updates: interactive human-in-the-loop approvals, live graph browsing updates, chat-like multi-turn sessions.

---

## 8. Tracing / Monitoring / Evals

### Tracing / Monitoring (Stage 3 & Stage 10)
- **LangSmith (Stage 3: tracing; Stage 10: evals scripts integration)**
- **Arize Phoenix (Stage 3)** — trace + retrieval debugging.
- **Helicone (Stage 3)** — LLM observability, cost/latency dashboards.

### Evals (Stage 10)
- **Self-hosted eval scripts (Stage 10)**
    - Metrics: schema pass rate, evidence coverage, verification fail rate, hallucination rate proxies, latency/cost.
        

---

## 9. Caching / Memory Mechanisms (Stage 11 + Stage 10)

### Caching
- **Semantic Cache (Stage 11)** — embedding-based query→answer caching.
- **Analysis cache (Stage 11)** — full-run caching keyed by inputs + doc checksums.
- **Web Caching (Stage 11)** — cached fetched sources with **1–7 day TTL**.
- **Context Caching / KV Cache (Stage 11)** — prompt+context reuse at runtime.
- **Cache invalidation strategy (Stage 11)** — TTL + checksum + version pinning for policies/models.

### Performance Techniques (Stage 11)
- **PagedAttention**
- **Quantization**
- **Grouped Query Attention (GQA)**
- **Context Window Compression**

### Memory Types (Stage 11)
- **Semantic Memory** — company summaries stored in vector DB.
- **Episodic Memory** — previous due diligence reports.
- **Graph Memory** — relationship graph in Neo4j.
- **company_profile_index (Stage 11)** — vector-based memory index.
- **company_node (Stage 11)** — Neo4j property-based memory store.

### Human Preference Memory (Stage 10)
- **Memory（记录基金偏好）** — stored preferences from human labeling loops and applied in ranking/scoring.

---

## 10. Guardrails / Security (Stage 8)
- **NVIDIA NeMo Guardrails**
- **Llama Guard**
- **Input Guard** — sanitize/deny disallowed intent and prompt injection.
- **Retrieval Guard** — source validation, allowlist/denylist, quality gates.
- **Output Guard** — redaction, policy compliance, uncertainty labeling.
- **Tool Permission** — per-tenant + per-run tool access control.
- **Sandbox Execution** — isolate web fetch, code execution, parsing.
- **Red Flag Detector** — flags sensitive decisions and high-risk compliance conclusions for review.

---

## 11. Other Core Agent Engineering Technologies / Environments

- **Agent Skills** — State of Art Ideas, define a agent skills
- Tools - For agent to use,
- **CLI or simple scripts (Stage 1)** — minimal user entrypoint for demos/testing.
- **JSON Mode / Structured Output (Stage 4)** — strict schemas for skills and nodes.
- **MCP Client / MCP Server (Stage 5)** — tool abstraction boundary.
- **AAIF agentic standards (MCP + AGENTS.md) (Stage 5)** — standard tool contracts, logging, schemas.
- **MCP streaming (Stage 5)** — streaming tool responses into state/UI.
- **parallel batching (Stage 5)** — batch retrieval/extraction for throughput.
- **Human-in-the-loop Edges (Stage 3)** — approval/review gates inside LangGraph.
- **Multi-tenant / multi-project concurrency (Stage 3)** — isolated runs, quotas, fair scheduling.
- **DSPy Program (Stage 9)** — optimization harness for skill modules.
- **DSPy Policies (Stage 9)** — policy definitions for routing/scoring.
- **structured prediction (Stage 9)**
- **policy optimization for skills (Stage 9)**
- **automatic module tuning (Stage 9)**
- **LLM → differentiable reasoning planner (Stage 9)**
- **Markdown / PDF export (Stage 12)** — final report generation.

---

## 12. Orchestration Design (LangGraph)

### 12.1 Core Nodes

1. **InputNode** (validate; Input Guard)
2. **DataAgentNode** (optional web fetch via requests/httpx + Web Caching + Sandbox)
3. **IngestDocsNode** (chunking + metadata)
4. **VectorIndexNode** (Chroma/FAISS)
5. **GraphBuildNode** (Neo4j upsert + provenance)
6. **Parallel Skill Node Group** (LangGraph parallel nodes)
    - applicability, obligations, risks, penalties, mitigations
7. **ICDebateNode** (o3/o3-mini; bull/bear/moderator)
8. **EvidenceVerificationNode** (retrieval guard + evidence checks)
9. **ComputeScoresNode**
10. **ReportNode** (Markdown + PDF)
11. **PersistMemoryNode** (Semantic/Episodic/Graph memory)
12. **EvalsNode** (self-hosted eval scripts + telemetry)
### 12.2 Human-in-the-loop Edges
- Approval edge after verification failures or red-flag triggers.
- Optional reviewer override to accept/reject uncertain claims.

---

## 13. Data Contracts (Structured Output)

### Skill Output Envelope (JSON Mode)
All Agent Skills must output:
- `items[]`
- `confidence`
- `evidence[]` (chunk_id and/or graph_path_id)
- `assumptions[]`
- `unknowns[]`

This enables downstream verification/scoring and UI drill-down.

---

## 14. GraphRAG Design (Neo4j + GraphQL)

### Graph Schema (minimal)
- Nodes: `Doc`, `Chunk`, `Law`, `Article`, `Obligation`, `Risk`, `Penalty`, `Concept`, `Product`, `Company`
- Edges: `HAS_CHUNK`, `MENTIONS`, `REQUIRES`, `TRIGGERS`, `SUBJECT_TO`, `LEADS_TO`, `MITIGATES`, `SUPPORTS`

### Provenance & Query
- Every derived node must link back to one/many `Chunk` nodes.
- Graph queries return a **graph_path_id** stored for replay and UI visualization.
- **neo4j-graphql** provides GraphQL endpoints for GraphRAG Visualization UI.

---

## 15. Async Execution + Real-time Communication

### Async Queue
- Celery workers execute runs; Redis stores queue state and event buffers.
### Streaming
- **SSE** for ordered progress feed (simple, robust).
- **WebSockets** for interactive HITL approvals and live exploration.

---

## 16. Observability & Tooling UX (Stage 12)
- **ReAct trace viewer**: connect to LangSmith/Phoenix traces.
- **LangGraph Studio**: visualize execution graph and checkpoints.
- **Agent logs viewer**: structured logs from FastAPI + workers.
- **GraphRAG Visualization UI**: graph explorer (Neo4j + GraphQL).

---

## 17. Deliverables

For each run:
- `report.md` (human-readable)
- `report.pdf` (export)
- `artifacts.json` (structured evidence, scores, debate transcript, verification results)
- trace links + logs + graph_path_id references for full auditability