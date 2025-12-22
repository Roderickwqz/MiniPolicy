# MiniPolicy 项目类图

## 核心类结构图

```mermaid
classDiagram
    %% 核心数据结构
    class Envelope {
        <<dataclass>>
        +envelope_type: EnvelopeType
        +run_id: str
        +node_id: str
        +step_id: str
        +ts_ms: int
        +graph_path_id: str
        +chunk_ids: List[str]
        +payload: Dict[str, Any]
        +error: Optional[Dict[str, Any]]
    }
    
    %% 状态管理
    class RunState {
        <<TypedDict>>
        +run_id: str
        +thread_id: str
        +user_input: Dict[str, Any]
        +chunks: Dict[str, Any]
        +vector_index: Dict[str, Any]
        +graph_build: Dict[str, Any]
        +envelopes: List[Dict[str, Any]]
        +report_path: Optional[str]
        +artifacts_path: Optional[str]
    }
    
    %% MCP 工具系统
    class MCPGateway {
        +fs: LocalFSTool
        +tools: Dict[str, ToolSpec]
        +call_tool()
        +_audit_tool_call()
    }
    
    class ToolSpec {
        <<dataclass>>
        +name: str
        +handler: Callable
        +schema: Dict[str, Any]
        +allow: bool
    }
    
    class LocalFSTool {
        +read_text()
        +write_text()
        +append_text()
        +read_json()
        +write_json()
        +append_json_record()
        -_resolve()
    }
    
    %% LangGraph 工作流
    class StateGraph {
        +add_node()
        +add_edge()
        +compile()
    }
    
    class SqliteSaver {
        +from_conn_string()
        +list()
    }
    
    %% 节点类
    class Node {
        <<abstract>>
        +execute(state: RunState): RunState
    }
    
    class InputNode {
        +node_input()
    }
    
    class IngestDocsNode {
        +node_ingest_docs()
    }
    
    class VectorIndexNode {
        +node_vector_index()
    }
    
    class GraphBuildNode {
        +node_graph_build()
    }
    
    class ParallelSkillsNode {
        +node_parallel_skills()
    }
    
    class DebateNode {
        +node_debate()
    }
    
    class VerifyNode {
        +node_verify()
    }
    
    class ScoreNode {
        +node_score()
    }
    
    class ReportNode {
        +node_report()
    }
    
    class EvalsNode {
        +node_evals()
    }
    
    %% CLI 入口
    class CLI {
        +load_json()
        +save_meta()
        +load_meta()
        +get_latest_checkpoint_id()
        +main()
    }
    
    %% 关系
    Envelope --> RunState : "存储在" envelopes
    MCPGateway --> ToolSpec : "管理" tools
    MCPGateway --> LocalFSTool : "使用" fs
    StateGraph --> Node : "包含多个" nodes
    SqliteSaver --> StateGraph : "提供" checkpointer
    CLI --> StateGraph : "调用" workflow
    CLI --> SqliteSaver : "使用" checkpointer
    
    InputNode --|> Node
    IngestDocsNode --|> Node
    VectorIndexNode --|> Node
    GraphBuildNode --|> Node
    ParallelSkillsNode --|> Node
    DebateNode --|> Node
    VerifyNode --|> Node
    ScoreNode --|> Node
    ReportNode --|> Node
    EvalsNode --|> Node
```

## MCP 工具层次结构

```mermaid
classDiagram
    %% 工具接口
    class Tool {
        <<interface>>
        +execute(args: Dict): Dict
    }
    
    %% 具体工具实现
    class LocalFSTool {
        +read_text()
        +write_text()
        +append_text()
        +read_json()
        +write_json()
        +append_json_record()
    }
    
    class PDFIngestTool {
        +ingest()
    }
    
    class VectorIndexTool {
        +upsert()
        +query()
    }
    
    class GraphStoreTool {
        +upsert()
        +query()
    }
    
    class ReportExportTool {
        +export()
    }
    
    %% 工具规范
    class ToolSpec {
        +name: str
        +handler: Callable
        +schema: Dict
        +allow: bool
    }
    
    %% 工具注册表
    class ToolRegistry {
        +tools: Dict[str, ToolSpec]
        +register()
        +get_spec()
    }
    
    %% 关系
    LocalFSTool ..|> Tool
    PDFIngestTool ..|> Tool
    VectorIndexTool ..|> Tool
    GraphStoreTool ..|> Tool
    ReportExportTool ..|> Tool
    
    ToolRegistry --> ToolSpec : "管理" tools
    MCPGateway --> ToolRegistry : "使用" registry
```

## 数据流类图

```mermaid
classDiagram
    %% 输入输出
    class InputData {
        +text: str
        +config: Dict
    }
    
    class OutputData {
        +report_path: str
        +artifacts_path: str
        +envelopes: List
    }
    
    %% 存储
    class ArtifactStorage {
        +save_report()
        +save_artifacts()
        +save_audit()
    }
    
    class CheckpointStorage {
        +save_state()
        +load_state()
        +list_checkpoints()
    }
    
    %% 处理流程
    InputData --> RunState : "初始化"
    RunState --> OutputData : "生成"
    OutputData --> ArtifactStorage : "保存"
    RunState --> CheckpointStorage : "持久化"
```

## 项目配置类图

```mermaid
classDiagram
    %% 项目配置
    class ProjectConfig {
        +PROJECT_ROOT: Path
        +ALLOWED_ROOTS: List[Path]
        +CHECKPOINT_PATH: str
        +ARTIFACTS_DIR: str
    }
    
    %% 环境常量
    class Environment {
        +DEBUG: bool
        +VERSION: str
        +API_BASE_URL: str
    }
    
    ProjectConfig ..> Environment : "包含"
```

## 完整的系统架构类图

```mermaid
classDiagram
    %% 分层架构
    package "表示层" {
        class WebUI
        class CLI
    }
    
    package "应用层" {
        class FastAPI
        class LangGraphRuntime
    }
    
    package "领域层" {
        class MCPGateway
        class ToolRegistry
        class Node
    }
    
    package "基础设施层" {
        class LocalFSTool
        class SqliteSaver
        class ArtifactStorage
    }
    
    %% 依赖关系
    WebUI --> FastAPI
    CLI --> FastAPI
    FastAPI --> LangGraphRuntime
    LangGraphRuntime --> MCPGateway
    MCPGateway --> ToolRegistry
    MCPGateway --> LocalFSTool
    LangGraphRuntime --> SqliteSaver
    Node --> ArtifactStorage
