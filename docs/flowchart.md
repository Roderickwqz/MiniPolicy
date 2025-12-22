# MiniPolicy 项目流程图

## 系统架构流程图

```mermaid
flowchart TD
    %% 用户入口
    UI[Web UI / CLI] --> API[FastAPI Backend]
    
    %% 核心工作流
    API --> LG[LangGraph Runtime]
    LG --> CP[SQLite Checkpointer]
    
    %% MCP 工具层
    LG --> MCP[Tool Permission Gateway]
    MCP --> FS[LocalFS Tool]
    MCP --> PDF[PDF Ingest Tool]
    MCP --> VEC[Vector Index Tool]
    MCP --> RET[Semantic Retrieve Tool]
    MCP --> GR[Graph Store Tool]
    MCP --> REP[Report Export Tool]
    
    %% 数据存储
    FS --> ART[app/artifacts/]
    FS --> RUNS[app/runs/]
    FS --> DOCS[docs/]
    
    VEC --> VDB[(Vector DB<br/>Chroma/FAISS)]
    GR --> N4J[(Neo4j Graph)]
    
    %% 节点流程
    subgraph Workflow["LangGraph 工作流"]
        IN[Input Node] --> INGEST[IngestDocs Node]
        INGEST --> VINDEX[VectorIndex Node]
        VINDEX --> GBUILD[GraphBuild Node]
        GBUILD --> SKILLS[ParallelSkills Node]
        SKILLS --> DEBATE[Debate Node]
        DEBATE --> VERIFY[Verify Node]
        VERIFY --> SCORE[Score Node]
        SCORE --> REPORT[Report Node]
        REPORT --> EVALS[Evals Node]
    end
    
    %% 连接
    LG --> Workflow
    
    %% 审计日志
    MCP --> AUDIT[audit.log]
    MCP --> ARTIFACTS[artifacts.json]
    
    %% 样式
    classDef gateway fill:#e1f5fe
    classDef node fill:#f3e5f5
    classDef tool fill:#e8f5e8
    classDef storage fill:#fff3e0
    
    class MCP gateway
    class IN,INGEST,VINDEX,GBUILD,SKILLS,DEBATE,VERIFY,SCORE,REPORT,EVALS node
    class FS,PDF,VEC,RET,GR,REP tool
    class ART,RUNS,DOCS,VDB,N4J storage
```

## 详细工作流流程图

```mermaid
sequenceDiagram
    participant User as 用户
    participant CLI as CLI/Run
    participant Graph as LangGraph
    participant Gateway as MCP Gateway
    participant Tools as Tools
    participant Storage as 存储
    
    User->>CLI: 启动任务
    CLI->>Graph: invoke(initial_state)
    Graph->>Graph: 检查 checkpoint
    
    loop 每个节点
        Graph->>Graph: 执行节点函数
        Graph->>Gateway: call_tool()
        Gateway->>Tools: 执行工具
        Tools->>Storage: 读写文件
        Storage-->>Tools: 返回结果
        Tools-->>Gateway: 返回输出
        Gateway-->>Graph: 返回结果
        Graph->>Graph: 记录 Envelope
    end
    
    Graph->>Graph: 生成报告
    Graph->>Gateway: 写入 artifacts
    Gateway->>Storage: 保存 report.md
    Gateway->>Storage: 保存 artifacts.json
    Graph-->>CLI: 返回最终状态
    CLI-->>User: 显示结果
```

## 数据流向图

```mermaid
flowchart LR
    subgraph Input["输入层"]
        DOC[文档输入] 
        USER[用户配置]
    end
    
    subgraph Process["处理层"]
        subgraph Ingest["摄取阶段"]
            PDF[PDF解析] --> CHUNK[分块]
        end
        
        subgraph Index["索引阶段"]
            VEC[向量索引] 
            GR[图索引]
        end
        
        subgraph Query["查询阶段"]
            RET[语义检索]
            GRQ[图查询]
        end
        
        subgraph Analyze["分析阶段"]
            SKILL[技能节点]
            DEBATE[辩论节点]
            VERIFY[验证节点]
        end
    end
    
    subgraph Output["输出层"]
        SCORE[评分]
        REPORT[报告生成]
        ARTIFACTS[工件保存]
    end
    
    DOC --> Ingest
    USER --> Process
    CHUNK --> Index
    VEC --> Query
    GR --> Query
    RET --> Analyze
    GRQ --> Analyze
    SKILL --> Output
    DEBATE --> Output
    VERIFY --> Output
```

## MCP 工具调用流程

```mermaid
flowchart TD
    CALL[节点调用工具] --> GATE[Gateway 接收]
    
    GATE --> CHECK{检查白名单}
    CHECK -->|允许| VALID{验证参数}
    CHECK -->|拒绝| DENY[返回错误]
    
    VALID -->|有效| EXEC[执行工具]
    VALID -->|无效| INVALID[返回错误]
    
    EXEC --> TOOL[具体工具实现]
    TOOL --> FS[文件系统操作]
    TOOL --> DB[数据库操作]
    
    FS -->|成功| OK[返回结果]
    FS -->|失败| ERR[返回错误]
    
    OK --> AUDIT[记录审计日志]
    ERR --> AUDIT
    
    AUDIT --> RETURN[返回给节点]
    
    %% 样式
    classDef success fill:#c8e6c9
    classDef error fill:#ffcdd2
    classDef process fill:#e3f2fd
    
    class OK,EXEC,AUDIT success
    class DENY,INVALID,ERR error
    class GATE,CHECK,VALID,TOOL,FS,RETURN process
