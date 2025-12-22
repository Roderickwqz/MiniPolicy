# app/mcp/tools/pdf_ingest.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple
import hashlib
import io
import os
import re
from dataclasses import dataclass

from app.mcp.contracts import ToolError, ToolResult
from llama_index.readers.file import PyPDFReader
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
LLAMAINDEX_AVAILABLE = True
from pypdf import PdfReader  # type: ignore
# except ImportError:
#     PyPDFReader = None
#     SentenceSplitter = None
#     TokenTextSplitter = None
#     LLAMAINDEX_AVAILABLE = False

# Fallback to pypdf if LlamaIndex not available
# try:

# except Exception:  # pragma: no cover - optional dependency
    # PdfReader = None


_DEFAULT_CHUNK_SIZE = 800
_DEFAULT_OVERLAP = 120


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tokenize_paragraphs(text: str) -> List[str]:
    """
    这个函数的作用是将文本按段落分割。
    """
    raw_blocks = re.split(r"\n\s*\n", text)
    return [block.strip() for block in raw_blocks if block.strip()]


def _chunk_deterministic(text: str, chunk_size: int, overlap: int) -> Iterable[Tuple[str, int, int]]:
    """
    这个函数的作用是将文本分割成固定大小的块，带有重叠部分。
    """
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
    在这个代码中，_chunk_semantic 的"语义"并不是指使用 LLM 模型进行语义理解，
    而是指基于文本的自然结构进行分割。
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
        # flush if adding this paragraph would exceed chunk_size
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


@dataclass
class _Page:
    number: int
    text: str


def _extract_pages(raw: bytes, pdf_path: str) -> List[_Page]:
    """
    Extract pages from PDF using LlamaIndex if available, fallback to pypdf.
    """
    # Try LlamaIndex first
    if LLAMAINDEX_AVAILABLE and PyPDFReader is not None:
        try:
            # Save to temp file for LlamaIndex
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            
            try:
                loader = PyPDFReader()
                documents = loader.load_data(file=tmp_path)
                
                pages: List[_Page] = []
                current_page = 1
                for doc in documents:
                    # Try to extract page number from metadata
                    page_num = doc.metadata.get("page_label") or doc.metadata.get("page_number")
                    if page_num:
                        try:
                            current_page = int(page_num)
                        except (ValueError, TypeError):
                            pass
                    
                    text = doc.get_content() or ""
                    if text.strip():
                        pages.append(_Page(number=current_page, text=text))
                        current_page += 1
                
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                
                if pages:
                    return pages
            except Exception:
                # Fallback to pypdf if LlamaIndex fails
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
    
    # Fallback to pypdf
    if PdfReader is None:
        text = raw.decode("utf-8", errors="ignore")
        return [_Page(number=1, text=text)]

    try:
        reader = PdfReader(io.BytesIO(raw))
    except Exception:
        text = raw.decode("utf-8", errors="ignore")
        return [_Page(number=1, text=text)]

    pages: List[_Page] = []
    for idx, page in enumerate(reader.pages):
        extracted = page.extract_text() or ""
        pages.append(_Page(number=idx + 1, text=extracted))
    return pages or [_Page(number=1, text="")]


def _chunk_page(
    doc_id: str,
    doc_hash: str,
    page: _Page,
    chunk_size: int,
    overlap: int,
    methods: List[str],
    pdf_path: str,
) -> List[Dict[str, Any]]:
    """
    Chunk a page using either custom logic or LlamaIndex splitters.
    Maintains stable chunk_id generation.
    """
    chunks: List[Dict[str, Any]] = []
    page_text = page.text or ""
    
    for method in methods:
        if method == "deterministic":
            # Use custom deterministic chunking (maintains exact same behavior)
            iterator = _chunk_deterministic(page_text, chunk_size, overlap)
        elif method == "semantic" and LLAMAINDEX_AVAILABLE and SentenceSplitter is not None:
            # Use LlamaIndex SentenceSplitter for semantic chunking
            try:
                splitter = SentenceSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                )
                # Create a simple document for splitting
                from llama_index.core import Document
                doc = Document(text=page_text, metadata={"page": page.number})
                nodes = splitter.get_nodes_from_documents([doc])
                
                # Convert nodes to our format
                iterator = []
                for node in nodes:
                    chunk_text = node.get_content()
                    # Estimate start/end positions (for compatibility)
                    start = page_text.find(chunk_text[:50]) if chunk_text else 0
                    end = start + len(chunk_text) if chunk_text else 0
                    iterator.append((chunk_text, start, end))
            except Exception:
                # Fallback to custom semantic chunking
                iterator = _chunk_semantic(page_text, chunk_size)
        else:
            # Fallback to custom semantic chunking
            iterator = _chunk_semantic(page_text, chunk_size)

        for idx, (chunk_text, start, end) in enumerate(iterator):
            # Maintain stable chunk_id generation (same format as before)
            chunk_hash = _sha256(f"{doc_hash}:{page.number}:{method}:{chunk_text}")
            chunk_id = f"{doc_id}::p{page.number:04d}::{method[:3]}::{chunk_hash[:10]}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "meta": {
                        "doc_id": doc_id,
                        "doc_hash": doc_hash,
                        "page": page.number,
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

    methods = (
        ["deterministic", "semantic"] if segmentation == "both" else [segmentation]
    )

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

    pages = _extract_pages(raw, pdf_path)
    chunks: List[Dict[str, Any]] = []
    for page in pages:
        page_chunks = _chunk_page(
            inferred_doc_id,
            doc_hash,
            page,
            chunk_size=chunk_size,
            overlap=overlap,
            methods=methods,
            pdf_path=pdf_path,
        )
        chunks.extend(page_chunks)

    return ToolResult(ok=True, tool_name="pdf_ingest_tool", data={"doc_id": inferred_doc_id, "chunks": chunks})
