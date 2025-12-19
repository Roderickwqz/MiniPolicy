## Phase 0 — “LangGraph 最小可运行骨架”（0.5 天）

**目标**：你在本地一条命令跑出 LangGraph，并看到节点流转、状态更新、checkpoint 写入。
**实现内容**
- LangGraph graph：按 Architect 12.1 的核心节点搭一个最小链路（先线性）  
    `Input → IngestDocs → VectorIndex → GraphBuild → ParallelSkills → Debate → Verify → Score → Report → Evals`
- Checkpointer：先用 SQLite / 本地文件 checkpointer（LangGraph 自带/或你自写），实现 **resume/replay** 的最小能力（FR-004）。
- CLI 入口：`python -m app.cli.run --input sample_run.json`

**验收**
- CLI 跑完输出：最终 state + report.md 路径 + artifacts.json 路径
- 中途 Ctrl+C，再跑同一个 run_id 能从 checkpoint 继续（哪怕只是“从下一个节点开始”）。


## Phase 1 — “MCP 工具层 + Tool Permission Gateway”（0.5 天）

**目标**：让所有节点都不直接 IO；任何读写都通过 gateway→tool，且有审计日志。
**实现内容（P0）**
- Gateway：allowlist + 参数校验 + 结构化审计日志（tool_name, args_hash, output_hash, latency, status）（FR-005）。
- MCP 工具接口（先本地实现，接口对齐即可）（FR-006）：
    - `pdf_ingest_tool`（读 PDF → chunks）
    - `vector_index_tool`（upsert chunks）
    - `semantic_retrieve_tool`（query → top_k）
    - `graph_upsert_tool`（写入图存储）
    - `graph_query_tool`（返回 graph_path_id）
    - `report_export_tool`（md/pdf/json）
- 工具“失败策略”：禁止静默失败；失败必须返回结构化 error，并写入 artifacts。

**验收**
- 代码里搜不到任何“节点直接读文件/写文件/直连 DB”的路径（除了 MCP tools）。
- 每次工具调用都会落一条 audit log，并能在 artifacts.json 看到 tool_call 记录。