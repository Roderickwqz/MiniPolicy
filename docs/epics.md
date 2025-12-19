# Epics & User Stories — Mini-Policy / Law IC Lab (Backlog v1)

## Global DoD (applies to all stories)

- Uses MCP tools for IO (no direct external IO in skills/nodes).
- Every user-visible claim/artifact is evidence-bound (`chunk_id` and/or `graph_path_id`) and replayable.
- Logs/trace/audit events emitted for key actions (for replay + debugging).

---

## EPIC-01 — Run Lifecycle + Async Orchestration (P0)

**Goal:** Create/track/resume a run via API; execute pipeline asynchronously with checkpoints and concurrency.
### STORY-01.1 — Create a run (POST /runs)

**As a** Builder/Demo Owner **I want** to create a run with `product_brief`, `jurisdictions[]`, `doc_registry[]` **so that** the pipeline can start.  
**AC**
- Given valid payload, `POST /runs` returns `run_id` and state is `RUN_STARTED`.
- Invalid payload returns 4xx with validation errors.  
    **Refs:** FR-001

### STORY-01.2 — Run status (GET /runs/{id})
**As a** Reviewer **I want** to query run status and basic metadata **so that** I can understand progress and current state.  
**AC**
- `GET /runs/{id}` returns run state, timestamps, selected docs, jurisdictions, and last checkpoint pointer.
- 404 for unknown run_id.  
    **Refs:** FR-001, FR-004

### STORY-01.3 — Async execution via Celery workers
**As a** Builder **I want** runs to execute asynchronously **so that** the API returns immediately and runs scale.  
**AC**
- API returns immediately after run creation.
- Worker executes pipeline and surfaces failures (with retry info).  
    **Refs:** FR-002

### STORY-01.4 — LangGraph checkpointing + resume
**As a** Engineer Reviewer **I want** checkpointed runs **so that** interruptions can resume safely.  
**AC**
- Checkpoints persist per node boundary.
- Resume continues from latest successful checkpoint.  
    **Refs:** FR-004, NFR-002

### STORY-01.5 — Multi-run concurrency
**As a** Builder **I want** multiple runs concurrently **so that** demos/experiments don’t block each other.  
**AC**
- Concurrent runs do not share state/evidence artifacts incorrectly.
- Basic isolation by run_id (and optional tenant/project key).  
    **Refs:** NFR-006

---

## EPIC-02 — Real-time Event Streaming + HITL (P0/P1)

**Goal:** Stream ordered events to UI; allow HITL approvals for flagged nodes, resuming from checkpoint.

### STORY-02.1 — SSE events endpoint
**As a** Reviewer **I want** real-time SSE events **so that** I can watch progress live.  
**AC**
- `GET /runs/{id}/events` streams ordered events including `NODE_STARTED/FINISHED`, `SKILL_OUTPUT`, `SCORE_UPDATED`, `REPORT_READY`.
- Client reconnect can continue (via last_event_id or replay buffer).  
    **Refs:** FR-003

### STORY-02.2 — WS approvals channel (HITL)
**As a** IC Reviewer **I want** to approve/deny flagged nodes **so that** uncertain claims can be reviewed.  
**AC**
- WS supports `APPROVAL_REQUESTED`, `APPROVAL_GRANTED/DENIED`, `RUN_RESUMED`.
- Approval decision is recorded in audit log and linked to checkpoint.  
    **Refs:** FR-003b, FR-004b

### STORY-02.3 — Review-required state + timeout policy
**As a** Engineer **I want** runs to pause in `REVIEW_REQUIRED` until approval or timeout **so that** the pipeline is safe and deterministic.  
**AC**
- Run transitions to `REVIEW_REQUIRED` on red-flag/verification failures.
- Timeout policy leads to explicit terminal state (e.g., `REVIEW_TIMED_OUT`) with report note.  
    **Refs:** FR-004b

---

## EPIC-03 — Ingestion + Vector RAG (P0)

**Goal:** Ingest PDFs into chunks with stable IDs; build vector index; retrieve evidence with strict JSON envelope.

### STORY-03.1 — PDF ingest MCP tool
**As a** Builder **I want** PDFs chunked with stable `chunk_id` **so that** evidence references are traceable.  
**AC**
- `pdf_ingest_tool` registers doc metadata and produces deterministic `chunk_id`.
- Rejects unsupported file types; logs reason.  
    **Refs:** FR-007, NFR-004

### STORY-03.2 — Vector indexing
**As a** Engineer **I want** a vector index from chunks **so that** semantic retrieval works offline-first.  
**AC**
- Build index (Chroma/FAISS) from ingested chunks.
- Index build emits timing/cost metrics.  
    **Refs:** FR-008, NFR-001

### STORY-03.3 — SemanticRetrieve skill strict envelope
**As a** Skill consumer node **I want** retrieval results in strict JSON envelope **so that** downstream verification can run reliably.  
**AC**
- Output includes `items[]`, `confidence`, `evidence[]` with `chunk_id`, `assumptions[]`, `unknowns[]`.
- Schema validation failure is surfaced and logged.  
    **Refs:** FR-008 + Data Contracts

---

## EPIC-04 — GraphRAG + Provenance + Graph Visualization API (P0/P1)

**Goal:** Build Neo4j graph with provenance; query returns `graph_path_id`; expose GraphQL for UI graph exploration.

### STORY-04.1 — Graph build with provenance links to chunks
**As a** Engineer **I want** derived graph nodes linked back to Chunk nodes **so that** graph evidence is explainable.  
**AC**
- GraphBuildNode upserts minimal schema entities and provenance edges.
- Every derived node stores origin `chunk_id`(s) or equivalent provenance relation.  
    **Refs:** FR-009

### STORY-04.2 — Graph query returns graph_path_id
**As a** Reviewer **I want** graph queries to return a replayable `graph_path_id` **so that** I can inspect paths in UI.  
**AC**
- Each graph query returns `graph_path_id` stored for replay.
- UI can fetch nodes/edges and provenance for a given path id.  
    **Refs:** FR-010

### STORY-04.3 — GraphQL endpoint for visualization
**As a** UI user **I want** GraphQL queries over Neo4j **so that** the graph explorer can browse nodes/edges.  
**AC**
- `/graphql` supports query of nodes/edges + provenance, drill-down by `graph_path_id`.
- Authorization rules enforced (at least by run scope).  
    **Refs:** FR-010b

---

## EPIC-05 — Parallel Analysis Skills + Mitigations (P0)

**Goal:** Run applicability/obligations/risks/penalties in parallel; produce mitigations with assumptions/unknowns and evidence binding.

### STORY-05.1 — Parallel skill execution + merge findings
**As a** Builder **I want** analysis skills to run in parallel **so that** latency is demo-acceptable.  
**AC**
- Skills execute concurrently and merge into a unified `findings` object.
- Partial failures are surfaced with degraded confidence (not silent).  
    **Refs:** FR-011, NFR-005

### STORY-05.2 — Mitigations with assumptions/unknowns
**As a** IC Reviewer **I want** mitigations to include assumptions/unknowns **so that** decisions reflect uncertainty.  
**AC**
- Mitigation items include `assumptions[]`, `unknowns[]` and evidence fields when possible.
- Missing evidence is explicitly labeled.  
    **Refs:** FR-012 + Data Contracts

---

## EPIC-06 — IC Debate + Moderation (P0)

**Goal:** Bull/Bear/Moderator debate produces disputed points, decision, and evidence-bound arguments.

### STORY-06.1 — Debate transcript + structured output
**As a** IC Reviewer **I want** an IC Debate output with bull/bear/disputed/decision **so that** tradeoffs are visible.  
**AC**
- Debate output includes: bull points, bear points, disputed points, decision, confidence.
- Each point includes evidence refs where possible (chunk/path).  
    **Refs:** FR-013

### STORY-06.2 — Moderator enforces evidence contract
**As a** Engineer Reviewer **I want** the moderator to reject unsupported claims **so that** outputs remain traceable.  
**AC**
- Moderator flags points missing evidence and requests re-retrieval or labels as uncertain.
- Flagged debate points can trigger HITL approval request.  
    **Refs:** Evidence contract + FR-004b

---

## EPIC-07 — Verification + Scoring (P0)

**Goal:** Evidence-Enforcer re-checks claims via RAG + GraphRAG validation; scoring decreases confidence on failures.

### STORY-07.1 — Verification pipeline (re-retrieval + path validation)
**As a** IC Reviewer **I want** claims labeled `verified/uncertain/failed` **so that** I can trust outputs appropriately.  
**AC**
- Verification step re-retrieves supporting chunks and validates graph paths.
- Emits per-claim status with evidence pointers.  
    **Refs:** FR-014

### STORY-07.2 — Multi-dimensional scoring + confidence adjustments
**As a** Builder **I want** a final score and confidence **so that** the run ends with a clear go/hold/avoid signal.  
**AC**
- Score includes subscores (configurable dimensions) + overall confidence.
- Confidence decreases with verification failures / weak evidence.  
    **Refs:** FR-015

### STORY-07.3 — Red-flag detector triggers review
**As a** Safety-conscious reviewer **I want** high-risk conclusions flagged **so that** I can intervene.  
**AC**
- Red-flag rules trigger `REVIEW_REQUIRED` and WS approval request.
- Flag event recorded in audit log.  
    **Refs:** Guardrails add-ons + FR-004b

---

## EPIC-08 — Reports + Exportable Artifacts (P0)

**Goal:** Generate Markdown + PDF report and artifacts.json including verification table and evidence refs.

### STORY-08.1 — Generate report.md with verification table
**As a** Reviewer **I want** a readable markdown report **so that** I can audit findings quickly.  
**AC**
- report.md includes: summary, findings, debate highlights, verification table, score, evidence references.
- Every claim includes chunk/path IDs or explicit “no evidence found”.  
    **Refs:** FR-017 + evidence contract

### STORY-08.2 — Generate report.pdf + artifacts.json
**As a** Builder **I want** PDF export and machine-readable artifacts **so that** the demo is shareable and automatable.  
**AC**
- report.pdf renders successfully and matches markdown content (at least sections & tables).
- artifacts.json includes key outputs: findings, evidence, scores, run metadata.  
    **Refs:** FR-017

---

## EPIC-09 — Web UI + Evidence/Trace/Graph Explorers (P1)

**Goal:** UI shows progress + debate timeline + scoring + evidence explorer + graph visualization and viewers.

### STORY-09.1 — Run dashboard (progress + timeline)
**As a** Demo Owner **I want** a run dashboard **so that** I can present the system live.  
**AC**
- UI shows: current node, progress, recent events, and debate timeline.
- Handles reconnect by reloading SSE history.  
    **Refs:** FR-018 + FR-003

### STORY-09.2 — Evidence explorer (chunk/path drill-down)
**As a** IC Reviewer **I want** to click a claim and see its `chunk_id` / `graph_path_id` evidence **so that** I can verify sources.  
**AC**
- Claim → list of evidence refs → view chunk text and graph path visualization.
- Missing evidence is clearly marked.  
    **Refs:** Evidence contract + FR-010/010b

### STORY-09.3 — Trace/log viewers integration
**As a** Engineer Reviewer **I want** trace/log viewers **so that** I can debug orchestration and tool calls.  
**AC**
- UI exposes links/panels for ReAct trace viewer, LangGraph Studio concepts, and agent logs viewer.
- Minimum: show structured log events per run.  
    **Refs:** FR-018, NFR-003

---

## EPIC-10 — Guardrails + Tool Permission Gateway (P0)

**Goal:** All tool calls go through permission + guardrails gateway; sandboxed web fetch optional; block/record unauthorized attempts.

### STORY-10.1 — Tool permission gateway enforcement
**As a** Security-minded engineer **I want** every tool call authorized **so that** unsafe capabilities are blocked.  
**AC**
- Unauthorized tool calls are blocked, logged, and visible in run output.
- Policy can be configured per-tenant/per-run.  
    **Refs:** FR-005

### STORY-10.2 — Input/Retrieval/Output guards
**As a** Demo Owner **I want** guardrails around inputs, retrieval sources, and outputs **so that** results are safe and policy-compliant.  
**AC**
- Input guard sanitizes prompt injection patterns.
- Retrieval guard enforces allowlist/quality gates.
- Output guard does redaction + uncertainty labeling.  
    **Refs:** FR-016

### STORY-10.3 — Sandbox execution for optional web fetch
**As a** Engineer **I want** optional web fetch to be sandboxed and cached **so that** offline-first remains safe.  
**AC**
- Web fetch is disabled by default; when enabled, runs in sandbox with caching TTL policy.
- All fetched content is tagged and auditable.  
    **Refs:** NFR-001 + Guardrails add-ons

---

## EPIC-11 — Observability + Evals (P0)

**Goal:** Store metrics per run; enable auditability + replay; expose eval trends and optimize quality loop.

### STORY-11.1 — Structured logs + audit_log + replay hooks
**As a** Engineer Reviewer **I want** auditable logs/traces **so that** every decision can be explained.  
**AC**
- Log key events: node transitions, tool calls, approvals, failures, export ready.
- Link logs to run_id and checkpoint ids.  
    **Refs:** NFR-003

### STORY-11.2 — Store eval metrics per run
**As a** Builder **I want** eval metrics stored **so that** I can track quality over time.  
**AC**
- Metrics include: schema pass rate, evidence coverage, verification fail rate, cost/latency.
- Accessible via API and/or included in artifacts.json.  
    **Refs:** FR-019

---

## EPIC-12 — Memory + Caching + DSPy Optimization (P1)

**Goal:** Improve rerun latency and consistency with memory/caches; optimize at least two modules with measurable metric improvement.

### STORY-12.1 — Memory persistence (semantic/episodic/graph)
**As a** Builder **I want** memory across runs **so that** reruns are faster and more consistent.  
**AC**
- Store semantic + episodic + graph memory keyed by project/tenant.
- Memory references remain evidence-bound and auditable.  
    **Refs:** FR-020

### STORY-12.2 — Explicit caching policy (semantic/analysis/context-KV)
**As a** Engineer **I want** explicit caches and TTL policies **so that** performance is predictable.  
**AC**
- Implement semantic cache, analysis cache, context/KV cache.
- Document invalidation strategy (by doc_registry hash or version).  
    **Refs:** FR-020b

### STORY-12.3 — DSPy optimization for 2 modules
**As a** Builder **I want** DSPy optimization **so that** at least one eval metric improves.  
**AC**
- Optimize ≥2 modules (e.g., AssessApplicability, ExtractObligations).
- Show before/after metric deltas in eval artifacts.  
    **Refs:** FR-021