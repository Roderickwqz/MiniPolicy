# Phase 2: Weaviate Schema 设计说明

## 1. vector_index_tool 的作用

`vector_index_tool` 是 RAG 流程中的关键步骤：

```
PDF → pdf_ingest_tool → chunks (内存)
                      ↓
chunks → vector_index_tool → Weaviate (向量数据库)
                            ↓
Weaviate → semantic_retrieve_tool → top_k chunks (语义检索)
```

**功能：**
- 将 PDF ingest 产生的 chunks 存储到 Weaviate 向量数据库
- 为每个 chunk 生成向量嵌入（通过 text2vec-openai）
- 存储 chunk 的元数据（doc_id, page, hash 等）
- 供后续的语义检索使用

## 2. Weaviate Schema 属性

当前设计的属性：

```python
properties = [
    {"name": "chunk_id", "dataType": ["string"]},  # 唯一标识符，用于 Neo4j 关联
    {"name": "text", "dataType": ["text"]},         # 文本内容（用于向量化）
    {"name": "doc_id", "dataType": ["string"]},     # 文档 ID（用于文档-块关系）
    {"name": "page", "dataType": ["int"]},         # 页码（用于定位）
    {"name": "hash", "dataType": ["string"]},       # 内容哈希（用于去重/验证）
    {"name": "meta", "dataType": ["text"]},         # JSON 元数据（完整信息）
]
```

## 3. 元数据完整性

从 `pdf_ingest_tool` 产生的 chunk 包含以下完整元数据（存储在 `meta` JSON 字段中）：

```python
{
    "doc_id": str,           # 文档标识
    "doc_hash": str,         # 文档哈希（用于验证）
    "page": int,             # 页码
    "start": int,            # 在页面中的起始位置
    "end": int,              # 在页面中的结束位置
    "hash": str,             # chunk 内容哈希
    "chunk_method": str,     # 分块方法（deterministic/semantic）
    "chunk_index": int,      # chunk 在页面中的索引
    "source_path": str,      # PDF 文件路径
}
```

## 4. 对 Neo4j GraphRAG 的兼容性

这些属性**足够**用于后续的 Neo4j 关系构建：

### 4.1 节点创建
- **Chunk 节点**：使用 `chunk_id` 作为唯一标识
- **Doc 节点**：使用 `doc_id` 作为唯一标识
- **Page 节点**：可以使用 `doc_id + page` 组合

### 4.2 关系创建
- `Doc -[:HAS_CHUNK]-> Chunk`：通过 `doc_id` 关联
- `Doc -[:HAS_PAGE]-> Page`：通过 `doc_id` 和 `page` 关联
- `Page -[:CONTAINS]-> Chunk`：通过 `doc_id`, `page`, `chunk_index` 关联
- `Chunk -[:NEXT_CHUNK]-> Chunk`：通过 `chunk_index` 顺序关联

### 4.3 溯源（Provenance）
- 每个 chunk 都有完整的 `meta` JSON，包含所有原始信息
- `hash` 字段可用于验证内容完整性
- `source_path` 可用于追溯原始文件

## 5. 如果需要扩展

如果后续需要更多字段，可以：

1. **添加新属性到 Weaviate schema**（需要重新创建 class 或使用动态 schema）
2. **在 `meta` JSON 中存储**（更灵活，无需修改 schema）
3. **在 Neo4j 中存储额外属性**（GraphRAG 阶段）

当前设计已经包含了 GraphRAG 所需的核心字段。
