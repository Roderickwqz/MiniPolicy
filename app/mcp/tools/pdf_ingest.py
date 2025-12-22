# app/mcp/tools/pdf_ingest.py
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import hashlib
import io
import os
import re
import tempfile

from app.mcp.contracts import ToolError, ToolResult
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from pypdf import PdfReader  # type: ignore

# Try to import OpenAI embedding model
try:
    from llama_index.embeddings.openai import OpenAIEmbedding

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    OpenAIEmbedding = None


_DEFAULT_CHUNK_SIZE = 800
_DEFAULT_OVERLAP = 120

# Cache for embedding model (lazy initialization)
_embed_model: Optional[Any] = None


def _get_embed_model():
    """
    Get or create OpenAI embedding model for semantic splitting.
    Returns None if OpenAI is not available or API key is not set.
    """
    global _embed_model
    if _embed_model is not None:
        return _embed_model

    if not _OPENAI_AVAILABLE:
        return None

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        _embed_model = OpenAIEmbedding()
        return _embed_model
    except Exception:
        return None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tokenize_paragraphs(text: str) -> List[str]:
    """将文本按段落分割（以空行作为段落分隔）。"""
    raw_blocks = re.split(r"\n\s*\n", text)
    return [block.strip() for block in raw_blocks if block.strip()]


def _chunk_deterministic(text: str, chunk_size: int, overlap: int) -> Iterable[Tuple[str, int, int]]:
    """将文本分割成固定大小的块，带有重叠部分。"""
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            yield chunk, start, end
        if end >= n:
            break
        start = max(end - overlap, 0)


def _chunk_semantic(text: str, chunk_size: int) -> Iterable[Tuple[str, int, int]]:
    """
    这里的“语义”是基于文本自然结构（段落）进行分割，而非 LLM 语义理解。
    """
    paragraphs = _tokenize_paragraphs(text)
    if not paragraphs:
        yield from _chunk_deterministic(text, chunk_size, overlap=0)
        return

    buffer: List[str] = []
    current_len = 0
    cursor = 0

    for para in paragraphs:
        para_len = len(para)
        if buffer and current_len + para_len > chunk_size:
            chunk_text = "\n\n".join(buffer).strip()
            end_cursor = cursor
            yield chunk_text, end_cursor - len(chunk_text), end_cursor
            buffer = []
            current_len = 0

        buffer.append(para)
        current_len += para_len + 2  # account for separator
        cursor += para_len + 2

    if buffer:
        chunk_text = "\n\n".join(buffer).strip()
        yield chunk_text, cursor - len(chunk_text), cursor


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except Exception:
        pass


def _load_pdf_documents(raw: bytes, pdf_path: str) -> List[Document]:
    """
    Load PDF and return List[Document] using LlamaIndex if available, fallback to pypdf.
    Each Document represents a page with metadata including page number.
    """
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        reader = SimpleDirectoryReader(
            input_files=[tmp_path],
            file_extractor={".pdf": PyMuPDFReader()},
        )
        documents = reader.load_data()

        current_page = 1
        for doc in documents:
            page_num = doc.metadata.get("page_label") or doc.metadata.get("page_number")
            if page_num:
                try:
                    current_page = int(page_num)
                except (ValueError, TypeError):
                    pass

            doc.metadata["page"] = current_page
            current_page += 1

        if documents:
            return documents

    except Exception:
        pass
    finally:
        if tmp_path:
            _safe_unlink(tmp_path)

    # Fallback: use pypdf to extract pages and create Documents
    try:
        reader = PdfReader(io.BytesIO(raw))
        documents: List[Document] = []
        for idx, page in enumerate(reader.pages):
            extracted = page.extract_text() or ""
            if extracted.strip():
                documents.append(Document(text=extracted, metadata={"page": idx + 1}))
        return documents if documents else [Document(text="", metadata={"page": 1})]
    except Exception:
        text = raw.decode("utf-8", errors="ignore")
        return [Document(text=text, metadata={"page": 1})]


def _process_nodes_to_chunks(
    nodes: List[Any],
    doc_id: str,
    doc_hash: str,
    method: str,
    pdf_path: str,
) -> List[Dict[str, Any]]:
    """
    Convert LlamaIndex nodes to our chunk format.
    Extracts page number from node metadata.
    """
    chunks: List[Dict[str, Any]] = []

    for idx, node in enumerate(nodes):
        chunk_text = node.get_content()
        if not chunk_text:
            continue

        page_num = node.metadata.get("page", 1)
        try:
            page_num = int(page_num)
        except (ValueError, TypeError):
            page_num = 1

        chunk_hash = _sha256(f"{doc_hash}:{page_num}:{method}:{chunk_text}")
        chunk_id = f"{doc_id}::p{page_num:04d}::{method[:3]}::{chunk_hash[:10]}"

        # 兼容：node 不携带原始 page_text，仍按你原实现给近似 start/end
        start = 0
        end = len(chunk_text)

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "meta": {
                    "doc_id": doc_id,
                    "doc_hash": doc_hash,
                    "page": page_num,
                    "start": start,
                    "end": end,
                    "hash": chunk_hash,
                    "chunk_method": method,
                    "chunk_index": idx,
                    "source_path": os.path.abspath(pdf_path),
                },
            }
        )

    return chunks


def _append_custom_chunk(
    chunks: List[Dict[str, Any]],
    *,
    doc_id: str,
    doc_hash: str,
    method: str,
    pdf_path: str,
    page_num: int,
    chunk_index: int,
    chunk_text: str,
    start: int,
    end: int,
) -> None:
    """统一封装自定义 chunk 的 append 模板，消除重复。"""
    if not chunk_text:
        return

    try:
        page_num_int = int(page_num)
    except (ValueError, TypeError):
        page_num_int = 1

    chunk_hash = _sha256(f"{doc_hash}:{page_num_int}:{method}:{chunk_text}")
    chunk_id = f"{doc_id}::p{page_num_int:04d}::{method[:3]}::{chunk_hash[:10]}"

    chunks.append(
        {
            "chunk_id": chunk_id,
            "text": chunk_text,
            "meta": {
                "doc_id": doc_id,
                "doc_hash": doc_hash,
                "page": page_num_int,
                "start": start,
                "end": end,
                "hash": chunk_hash,
                "chunk_method": method,
                "chunk_index": chunk_index,
                "source_path": os.path.abspath(pdf_path),
            },
        }
    )


def _chunks_via_custom_chunker(
    documents: List[Document],
    *,
    doc_id: str,
    doc_hash: str,
    method: str,
    pdf_path: str,
    chunker: Callable[..., Iterable[Tuple[str, int, int]]],
    chunker_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    将 documents 按页取出文本，交给自定义 chunker，统一构造 chunks。
    这是原代码中重复最多的 for doc + enumerate(chunker(...)) 逻辑的收敛点。
    """
    chunks: List[Dict[str, Any]] = []
    for doc in documents:
        page_num = doc.metadata.get("page", 1)
        try:
            page_num_int = int(page_num)
        except (ValueError, TypeError):
            page_num_int = 1

        page_text = doc.get_content() or ""
        for idx, (chunk_text, start, end) in enumerate(chunker(page_text, **chunker_kwargs)):
            _append_custom_chunk(
                chunks,
                doc_id=doc_id,
                doc_hash=doc_hash,
                method=method,
                pdf_path=pdf_path,
                page_num=page_num_int,
                chunk_index=idx,
                chunk_text=chunk_text,
                start=start,
                end=end,
            )
    return chunks


def _chunks_via_sentence_splitter(
    documents: List[Document],
    *,
    doc_id: str,
    doc_hash: str,
    method: str,
    pdf_path: str,
    chunk_size: int,
    overlap: int,
) -> List[Dict[str, Any]]:
    """SentenceSplitter → nodes → chunks 的统一封装，消除重复。"""
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    return _process_nodes_to_chunks(nodes, doc_id, doc_hash, method, pdf_path)


def _chunks_via_semantic_splitter(
    documents: List[Document],
    *,
    doc_id: str,
    doc_hash: str,
    method: str,
    pdf_path: str,
    embed_model: Any,
) -> List[Dict[str, Any]]:
    """SemanticSplitterNodeParser → nodes → chunks 的统一封装。"""
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    return _process_nodes_to_chunks(nodes, doc_id, doc_hash, method, pdf_path)


def pdf_ingest_tool(args: Dict[str, Any]) -> ToolResult:
    """
    args:
      - pdf_path: str
      - chunk_size: int (optional)
      - overlap: int (optional)
      - segmentation: "deterministic" | "semantic" | "both" (optional)
      - doc_id: str (optional)
    returns:
      - chunks: [{chunk_id, text, meta}]
    """
    pdf_path = args["pdf_path"]
    chunk_size = max(64, int(args.get("chunk_size") or _DEFAULT_CHUNK_SIZE))
    overlap = max(0, int(args.get("overlap") or _DEFAULT_OVERLAP))
    segmentation = (args.get("segmentation") or "deterministic").lower()
    doc_id = args.get("doc_id")

    if segmentation not in {"deterministic", "semantic", "both"}:
        return ToolResult(
            ok=False,
            tool_name="pdf_ingest_tool",
            error=ToolError(
                code="VALIDATION_ERROR",
                message=f"Unsupported segmentation '{segmentation}'",
                details={"expected": ["deterministic", "semantic", "both"]},
            ),
        )

    methods = ["deterministic", "semantic"] if segmentation == "both" else [segmentation]

    try:
        with open(pdf_path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return ToolResult(
            ok=False,
            tool_name="pdf_ingest_tool",
            error=ToolError(code="NOT_FOUND", message="PDF not found", details={"pdf_path": pdf_path}),
        )

    doc_hash = _file_hash(raw)
    inferred_doc_id = doc_id or f"doc::{doc_hash[:12]}"

    documents = _load_pdf_documents(raw, pdf_path)
    if not documents:
        return ToolResult(
            ok=False,
            tool_name="pdf_ingest_tool",
            error=ToolError(code="PROCESSING_ERROR", message="Failed to extract text from PDF"),
        )

    chunks: List[Dict[str, Any]] = []

    for method in methods:
        if method == "deterministic":
            # 优先 SentenceSplitter，失败则 custom deterministic
            try:
                chunks.extend(
                    _chunks_via_sentence_splitter(
                        documents,
                        doc_id=inferred_doc_id,
                        doc_hash=doc_hash,
                        method=method,
                        pdf_path=pdf_path,
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                )
            except Exception:
                chunks.extend(
                    _chunks_via_custom_chunker(
                        documents,
                        doc_id=inferred_doc_id,
                        doc_hash=doc_hash,
                        method=method,
                        pdf_path=pdf_path,
                        chunker=_chunk_deterministic,
                        chunker_kwargs={"chunk_size": chunk_size, "overlap": overlap},
                    )
                )

        elif method == "semantic":
            # 语义分割：按优先级依次尝试（若 embed 可用：SemanticSplitter → SentenceSplitter → custom semantic）
            embed_model = _get_embed_model()

            # 1) SemanticSplitter（若可用）
            if embed_model is not None:
                try:
                    chunks.extend(
                        _chunks_via_semantic_splitter(
                            documents,
                            doc_id=inferred_doc_id,
                            doc_hash=doc_hash,
                            method=method,
                            pdf_path=pdf_path,
                            embed_model=embed_model,
                        )
                    )
                    continue
                except Exception:
                    pass

            # 2) SentenceSplitter fallback
            try:
                chunks.extend(
                    _chunks_via_sentence_splitter(
                        documents,
                        doc_id=inferred_doc_id,
                        doc_hash=doc_hash,
                        method=method,
                        pdf_path=pdf_path,
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                )
            except Exception:
                # 3) custom semantic fallback
                chunks.extend(
                    _chunks_via_custom_chunker(
                        documents,
                        doc_id=inferred_doc_id,
                        doc_hash=doc_hash,
                        method=method,
                        pdf_path=pdf_path,
                        chunker=_chunk_semantic,
                        chunker_kwargs={"chunk_size": chunk_size},
                    )
                )

        else:
            # Unknown method, skip
            continue

    return ToolResult(ok=True, tool_name="pdf_ingest_tool", data={"doc_id": inferred_doc_id, "chunks": chunks})
