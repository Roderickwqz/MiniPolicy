# LlamaIndex Index Types: VectorStoreIndex vs SummaryIndex

## Overview

LlamaIndex provides two primary indexing structures, each designed for different use cases.

## VectorStoreIndex

**Purpose**: Semantic similarity search and retrieval

**How it works**:
- Divides documents into smaller segments (nodes)
- Generates vector embeddings for each node
- During query, retrieves top-k most relevant nodes based on semantic similarity

**Use cases**:
- ✅ **Semantic search**: Find information that semantically matches the query
- ✅ **Large datasets**: Efficiently narrows search space to most relevant nodes
- ✅ **RAG applications**: Retrieve relevant context for LLM generation

**Example**:
```python
from llama_index.core import VectorStoreIndex

# Build index from documents
index = VectorStoreIndex.from_documents(documents)

# Query using retriever
retriever = index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve(query)

# Or use query engine (retriever + LLM)
query_engine = index.as_query_engine()
response = query_engine.query("What are the requirements?")
```

## SummaryIndex

**Purpose**: Generate comprehensive summaries of entire dataset

**How it works**:
- Stores all nodes without vectorization
- Upon querying, sends ALL nodes to the language model
- LLM generates a response based on the entire dataset

**Use cases**:
- ✅ **Summarization**: Generate holistic summary of entire dataset
- ✅ **Small datasets**: When entire dataset fits in LLM context window
- ❌ **NOT for retrieval**: Not designed for finding specific information

**Example**:
```python
from llama_index.core import SummaryIndex

# Build index from documents
index = SummaryIndex.from_documents(documents)

# Query using query engine (sends all nodes to LLM)
query_engine = index.as_query_engine(response_mode="tree_summarize")
response = query_engine.query("Provide a summary of the collection.")
```

## Why We Only Use VectorStoreIndex

For our Phase 2 RAG implementation:

1. **Our goal**: Semantic retrieval of relevant chunks (not summarization)
2. **Our data**: PDFs with many chunks - too large to send all to LLM
3. **Our workflow**: 
   - Store chunks in Weaviate (vector_index_tool)
   - Retrieve top_k relevant chunks (semantic_retrieve_tool)
   - Use retrieved chunks as context for downstream tasks

**SummaryIndex would NOT work** because:
- It requires sending ALL chunks to LLM (expensive, slow)
- It's designed for summarization, not retrieval
- Our use case is semantic search, not holistic summarization

## Architecture Flow

```
PDF → pdf_ingest_tool → chunks
                      ↓
chunks → vector_index_tool → Weaviate (VectorStoreIndex)
                            ↓
Weaviate → semantic_retrieve_tool → top_k chunks
         (uses VectorStoreIndex.as_retriever())
```

## When to Use Each

| Use Case | Index Type | Reason |
|----------|-----------|--------|
| Semantic search / RAG | **VectorStoreIndex** | Efficient retrieval of relevant chunks |
| Generate dataset summary | **SummaryIndex** | Need holistic view of all data |
| Large datasets | **VectorStoreIndex** | Can't fit all in context |
| Small datasets + summary | **SummaryIndex** | Can fit all in context |

## Conclusion

For Phase 2, we **only need VectorStoreIndex** because:
- ✅ We're doing semantic retrieval (not summarization)
- ✅ We have large datasets (many PDF chunks)
- ✅ We need efficient top-k retrieval
- ❌ We don't need to summarize entire dataset at once
