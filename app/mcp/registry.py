# app/mcp/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from app.mcp.tools.local_fs import LocalFSTool
from app.mcp.tools.pdf_ingest import pdf_ingest_tool
from app.mcp.tools.vector_index import vector_index_tool
from app.mcp.tools.semantic_retrieve import semantic_retrieve_tool
from app.mcp.tools.graph_store import graph_upsert_tool, graph_query_tool
from app.mcp.tools.report_export import report_export_tool

# Import centralized configuration for validation
from app.mcp.config import get_config


@dataclass(frozen=True)
class ToolSpec:
    name: str
    handler: Callable[[Dict[str, Any]], Any]   # 入参统一是 dict
    schema: Dict[str, Any]                     # P0 schema: required + properties(type)
    allow: bool = True


def build_registry(fs: LocalFSTool) -> Dict[str, ToolSpec]:
    """
    Phase1 allowlist：显式注册（推荐）
    你后续新增工具，就在这里加一条 ToolSpec。
    """
    return {
        # ---------- FS ----------
        "fs.read_text": ToolSpec(
            name="fs.read_text",
            handler=lambda a: fs.read_text(**a),
            schema={"required": ["path"], "properties": {"path": {"type": "string"}}},
        ),
        "fs.write_text": ToolSpec(
            name="fs.write_text",
            handler=lambda a: fs.write_text(**a),
            schema={
                "required": ["path", "text"],
                "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
            },
        ),
        "fs.read_json": ToolSpec(
            name="fs.read_json",
            handler=lambda a: fs.read_json(**a),
            schema={"required": ["path"], "properties": {"path": {"type": "string"}}},
        ),
        "fs.write_json": ToolSpec(
            name="fs.write_json",
            handler=lambda a: fs.write_json(**a),
            schema={
                "required": ["path", "data"],
                "properties": {"path": {"type": "string"}, "data": {"type": "object"}},
            },
        ),

        # 你刚加的两个接口：用于 Phase1 审计与 artifacts 追加
        "fs.append_text": ToolSpec(
            name="fs.append_text",
            handler=lambda a: fs.append_text(**a),
            schema={
                "required": ["path", "text"],
                "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
            },
        ),
        "fs.append_json_record": ToolSpec(
            name="fs.append_json_record",
            handler=lambda a: fs.append_json_record(**a),
            schema={
                "required": ["path", "record"],
                "properties": {"path": {"type": "string"}, "record": {"type": "object"}},
            },
        ),

        # ---------- Phase1 MCP interface placeholders ----------
        # 先占位“接口对齐”：后面你再把 handler 指向真实实现
        "pdf.ingest": ToolSpec(
            name="pdf.ingest",
            handler=pdf_ingest_tool,
            schema={
                "required": ["pdf_path"],
                "properties": {
                    "pdf_path": {"type": "string"},
                    "doc_id": {"type": "string"},
                    "chunk_size": {"type": "integer"},
                    "overlap": {"type": "integer"},
                    "segmentation": {"type": "string"},
                },
            },
        ),
        "vector.upsert": ToolSpec(
            name="vector.upsert",
            handler=vector_index_tool,
            schema={
                "required": ["index_name", "chunks"],
                "properties": {"index_name": {"type": "string"}, "chunks": {"type": "array"}},
            },
        ),
        "vector.query": ToolSpec(
            name="vector.query",
            handler=semantic_retrieve_tool,
            schema={
                "required": ["index_name", "query", "top_k"],
                "properties": {
                    "index_name": {"type": "string"},
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
            },
        ),
        "graph.upsert": ToolSpec(
            name="graph.upsert",
            handler=graph_upsert_tool,
            schema={
                "required": ["nodes", "edges"],
                "properties": {"nodes": {"type": "array"}, "edges": {"type": "array"}},
            },
        ),
        "graph.query": ToolSpec(
            name="graph.query",
            handler=graph_query_tool,
            schema={"required": ["query"], "properties": {"query": {"type": "string"}}},
        ),
        "report.export": ToolSpec(
            name="report.export",
            handler=report_export_tool,
            schema={
                "required": ["run_dir", "format", "content", "filename"],
                "properties": {
                    "run_dir": {"type": "string"},
                    "format": {"type": "string"},
                    "content": {"type": "object"},
                    "filename": {"type": "string"},
                },
            },
        ),
    }
