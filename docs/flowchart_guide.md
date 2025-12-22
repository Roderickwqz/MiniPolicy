# MiniPolicy Phase 0-1 流程图指南

## Phase 0-1 核心流程

```mermaid
flowchart TD
    %% 用户入口
    CLI[CLI Run] --> LG[LangGraph Runtime]
    
    %% 工作流节点
    LG --> INPUT[Input Node]
    INPUT --> INGEST[IngestDocs Node]
    INGEST --> VINDEX[VectorIndex Node]
    VINDEX --> GBUILD[GraphBuild Node]
    GBUILD --> SKILLS[ParallelSkills Node]
    SKILLS --> DEBATE[Debate Node]
    DEBATE --> VERIFY[Verify Node]
    VERIFY --> SCORE[Score Node]
    SCORE --> REPORT[Report Node]
    REPORT --> EVALS[Evals Node]
    
    %% 状态管理
    LG --> CP[SQLite Checkpointer]
    CP --> CP_STATE[Checkpoint State]
    
    %% MCP Gateway 和 Tools
    LG --> MCP[MCP Gateway]
    MCP --> FS[LocalFS Tool]
    
    %% Artifact 和 Runs 目录
    FS --> ART[app/artifacts/]
    FS --> RUNS[app/runs/]
    
    %% Artifact 子目录（修复：含 {run_id} 的标签改为带引号）
    ART --> RUN_DIR["app/artifacts/{run_id}/"]
    RUN_DIR --> REPORT_MD[report.md]
    RUN_DIR --> ARTIFACTS_JSON[artifacts.json]
    RUN_DIR --> AUDIT_LOG[audit.log]
    RUN_DIR --> ENVELOPES_JSON[envelopes.json]
    
    %% Runs 子目录
    RUNS --> META[app/runs/meta/]
    RUNS --> CHECKPOINTS[app/runs/checkpoints.sqlite]
    META --> META_JSON[run_id.json]
    
    %% 审计流程
    MCP --> AUDIT_LOG
    MCP --> ARTIFACTS_JSON
    
    %% 样式定义
    classDef gateway fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef node fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef tool fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef storage fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef artifact fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    
    class MCP gateway
    class INPUT,INGEST,VINDEX,GBUILD,SKILLS,DEBATE,VERIFY,SCORE,REPORT,EVALS node
    class FS tool
    class ART,RUNS,CHECKPOINTS,REPORT_MD,ARTIFACTS_JSON,AUDIT_LOG,ENVELOPES_JSON,CP_STATE storage
    class RUN_DIR,META,REPORT_MD,ARTIFACTS_JSON,AUDIT_LOG,ENVELOPES_JSON,META_JSON artifact

```

## Phase 0-1 数据流详细图

```mermaid
sequenceDiagram
    participant CLI as CLI
    participant LG as LangGraph
    participant CP as Checkpointer
    participant MCP as MCP Gateway
    participant FS as LocalFS Tool
    participant ART as Artifacts
    participant RUNS as Runs
    
    %% 初始化
    CLI->>LG: invoke(initial_state)
    LG->>CP: 检查 checkpoint
    Note over LG,CP: 从 SQLite 读取历史状态
    
    %% 节点执行流程
    loop 每个节点
        LG->>LG: 执行节点函数
        LG->>MCP: call_tool()
        MCP->>FS: 执行文件操作
        FS->>ART: 写入文件
        FS-->>MCP: 返回结果
        MCP-->>LG: 返回工具结果
        LG->>LG: 记录 Envelope
        Note over LG: 每个节点都会生成 Envelope
    end
    
    %% Report 节点特殊处理
    LG->>MCP: call_tool(fs.write_text)
    MCP->>FS: 写入 report.md
    FS->>ART: app/artifacts/{run_id}/report.md
    FS-->>MCP: 写入成功
    MCP-->>LG: 返回
    
    LG->>MCP: call_tool(fs.write_json)
    MCP->>FS: 写入 artifacts.json
    FS->>ART: app/artifacts/{run_id}/artifacts.json
    FS-->>MCP: 写入成功
    MCP-->>LG: 返回
    
    %% 审计日志
    MCP->>FS: 写入 audit.log
    FS->>ART: app/artifacts/{run_id}/audit.log
    FS-->>MCP: 写入成功
    
    %% Checkpoint 保存
    LG->>CP: 保存状态
    CP->>RUNS: checkpoints.sqlite
    CP-->>LG: 保存成功
    
    %% 元数据保存
    LG->>FS: 写入 meta.json
    FS->>RUNS: app/runs/meta/{run_id}.json
    FS-->>LG: 写入成功
    
    LG-->>CLI: 返回最终状态
    CLI-->>用户: 显示结果
```

## Artifact 目录结构

```mermaid
graph TD
    ART[app/artifacts/]
    
    subgraph RunArtifacts["每个 run_id 的 artifacts"]
        RUN_DIR["app/artifacts/{run_id}/"]
        REPORT_MD[report.md]
        ARTIFACTS_JSON[artifacts.json]
        AUDIT_LOG[audit.log]
        ENVELOPES_JSON[envelopes.json]
    end
    
    subgraph RunFiles["Artifacts 内容"]
        ENVELOPES_JSON --> ENVELOPES[Envelope 列表]
        ARTIFACTS_JSON --> TOOL_CALLS[Tool Call 记录]
        AUDIT_LOG --> AUDIT_RECORDS[审计日志行]
        REPORT_MD --> REPORT_CONTENT[Markdown 报告]
    end
    
    ART --> RUN_DIR
    RUN_DIR --> REPORT_MD
    RUN_DIR --> ARTIFACTS_JSON
    RUN_DIR --> AUDIT_LOG
    RUN_DIR --> ENVELOPES_JSON
    
    %% 样式
    classDef artifact fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef content fill:#e8eaf6,stroke:#283593,stroke-width:1px
    
    class RUN_DIR,REPORT_MD,ARTIFACTS_JSON,AUDIT_LOG,ENVELOPES_JSON artifact
    class ENVELOPES,TOOL_CALLS,AUDIT_RECORDS,REPORT_CONTENT content

```

## Runs 目录结构

```mermaid
graph TD
    RUNS[app/runs/]
    
    subgraph RunStorage["运行时存储"]
        META[app/runs/meta/]
        CHECKPOINTS[app/runs/checkpoints.sqlite]
    end
    
    subgraph MetaFiles["Meta 文件"]
        META_JSON[run_id.json]
        META_INFO[run_id 和 thread_id]
        META_DESC[运行元数据：run_id 和 thread_id 的映射]
        META_EX[示例：{"run_id": "run_5f80c3a21e", "thread_id": "run_5f80c3a21e"}]
    end
    
    subgraph CheckpointData["Checkpoint 数据"]
        CP_DB[SQLite 数据库]
        CP_STATES[状态记录]
        CP_CONFIGS[配置信息]
    end
    
    RUNS --> META
    RUNS --> CHECKPOINTS
    META --> META_JSON
    META_JSON --> META_INFO
    CHECKPOINTS --> CP_DB
    CP_DB --> CP_STATES
    CP_DB --> CP_CONFIGS
    
    %% 样式
    classDef storage fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef data fill:#e0f2f1,stroke:#004d40,stroke-width:1px
    
    class META,CHECKPOINTS,CP_DB storage
    class META_INFO,CP_STATES,CP_CONFIGS data
```

## 审计流程详细图

```mermaid
flowchart TD
    NODE[节点调用工具] --> GATE[MCP Gateway]
    
    GATE --> CHECK{检查白名单}
    CHECK -->|允许| VALID{验证参数}
    CHECK -->|拒绝| DENY[返回错误]
    
    VALID -->|有效| EXEC[执行工具]
    VALID -->|无效| INVALID[返回错误]
    
    EXEC --> FS[LocalFS Tool]
    FS --> FILE[文件操作]
    
    FILE -->|成功| OK[返回结果]
    FILE -->|失败| ERR[返回错误]
    
    OK --> AUDIT[记录审计日志]
    ERR --> AUDIT
    
    AUDIT --> WRITE_LOG[写入 audit.log]
    AUDIT --> WRITE_ARTIFACTS[写入 artifacts.json]
    
    WRITE_LOG --> LOG_FILE["app/artifacts/{run_id}/audit.log"]
    WRITE_ARTIFACTS --> ARTIFACTS_FILE["app/artifacts/{run_id}/artifacts.json"]
    
    LOG_FILE --> RETURN[返回给节点]
    ARTIFACTS_FILE --> RETURN
    
    %% 样式
    classDef success fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    classDef error fill:#ffcdd2,stroke:#c62828,stroke-width:2px
    classDef process fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef storage fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class OK,AUDIT,SUCCESS success
    class DENY,INVALID,ERR error
    class GATE,CHECK,VALID,EXEC,FS,FILE,RETURN process
    class LOG_FILE,ARTIFACTS_FILE storage

```

## Phase 0-1 完整执行流程

```mermaid
flowchart LR
    subgraph User["用户层"]
        CLI[CLI 命令]
    end
    
    subgraph Runtime["运行时层"]
        LG[LangGraph Runtime]
        CP[SQLite Checkpointer]
    end
    
    subgraph Control["控制层"]
        MCP[MCP Gateway]
        FS[LocalFS Tool]
    end
    
    subgraph Storage["存储层"]
        ART[Artifacts 目录]
        RUNS[Runs 目录]
    end
    
    subgraph Artifacts["Artifacts 内容"]
        REPORT[report.md]
        ARTIFACTS[artifacts.json]
        AUDIT[audit.log]
        ENVELOPES[envelopes.json]
    end
    
    subgraph Runs["Runs 内容"]
        META[meta/]
        CHECKPOINT[checkpoints.sqlite]
    end
    
    CLI --> LG
    LG --> CP
    LG --> MCP
    MCP --> FS
    FS --> ART
    FS --> RUNS
    
    ART --> REPORT
    ART --> ARTIFACTS
    ART --> AUDIT
    ART --> ENVELOPES
    
    RUNS --> META
    RUNS --> CHECKPOINT
    
    %% 样式
    classDef user fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
    classDef runtime fill:#e0f2f1,stroke:#009688,stroke-width:2px
    classDef control fill:#e1f5fe,stroke:#039be5,stroke-width:2px
    classDef storage fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef artifact fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class CLI user
    class LG,CP runtime
    class MCP,FS control
    class ART,RUNS storage
    class REPORT,ARTIFACTS,AUDIT,ENVELOPES artifact
