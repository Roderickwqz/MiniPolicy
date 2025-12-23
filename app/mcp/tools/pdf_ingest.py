# app/mcp/tools/pdf_ingest.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable
import hashlib
import io
import os
import re
import tempfile
import threading

from app.mcp.contracts import ToolError, ToolResult

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from pypdf import PdfReader  # type: ignore


# ----------------------------
# Optional OpenAI embedding
# ----------------------------
try:
    from llama_index.embeddings.openai import OpenAIEmbedding

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    OpenAIEmbedding = None  # type: ignore[misc]

# Import centralized configuration
from app.mcp.config import get_openai_api_key


_DEFAULT_CHUNK_SIZE = 800
_DEFAULT_OVERLAP = 120

# Global cache (thread-safe) for embedding model
_embed_model: Optional[Any] = None
_embed_lock = threading.Lock()


# ----------------------------
# Hash / normalize helpers
# ----------------------------
def _sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except Exception:
        pass


def _normalize_text_for_id(text: str) -> str:
    """
    Used ONLY for stable content_id:
    - strip
    - collapse any whitespace to single spaces
    """
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _approx_tokens(text: str) -> int:
    """
    Lightweight heuristic:
    - English-ish: ~4 chars per token
    - CJK: tends to be ~1.5-2 chars/token, but without heavy deps we keep a safe heuristic.
    This is for metadata/observability only.
    """
    t = (text or "").strip()
    if not t:
        return 0
    # If contains lots of CJK, reduce divisor a bit
    cjk = re.findall(r"[\u4e00-\u9fff]", t)
    if len(cjk) / max(len(t), 1) > 0.15:
        return max(1, int(len(t) / 2.0))
    return max(1, int(len(t) / 4.0))


# ----------------------------
# Text cleaning (P2)
# ----------------------------
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
_MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")


def _merge_hyphenation(text: str) -> str:
    # "com-\npliance" -> "compliance"
    return _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)


def _clean_text(
    text: str,
    *,
    merge_hyphenation: bool = True,
    collapse_blank_lines: bool = True,
) -> str:
    t = text or ""
    if merge_hyphenation:
        t = _merge_hyphenation(t)
    if collapse_blank_lines:
        t = _MULTIPLE_NEWLINES_RE.sub("\n\n", t)
    return t


def _extract_lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _remove_headers_footers(
    documents: List[Document],
    *,
    header_lines: int = 2,
    footer_lines: int = 2,
    min_repetition_ratio: float = 0.35,
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Heuristic removal of repeated headers/footers across pages.
    - Collect top N and bottom N non-empty lines on each page
    - Remove lines that repeat across enough pages
    """
    page_texts: List[str] = [doc.get_content() or "" for doc in documents]
    pages = len(page_texts)
    if pages <= 2:
        return documents, {"enabled": True, "removed": False, "reason": "too_few_pages"}

    header_counts: Dict[str, int] = {}
    footer_counts: Dict[str, int] = {}

    page_headers: List[List[str]] = []
    page_footers: List[List[str]] = []

    for txt in page_texts:
        lines = _extract_lines(txt)
        hdr = lines[:header_lines] if header_lines > 0 else []
        ftr = lines[-footer_lines:] if footer_lines > 0 else []
        page_headers.append(hdr)
        page_footers.append(ftr)
        for ln in hdr:
            header_counts[ln] = header_counts.get(ln, 0) + 1
        for ln in ftr:
            footer_counts[ln] = footer_counts.get(ln, 0) + 1

    threshold = max(2, int(pages * min_repetition_ratio))

    frequent_headers = {ln for ln, c in header_counts.items() if c >= threshold}
    frequent_footers = {ln for ln, c in footer_counts.items() if c >= threshold}

    if not frequent_headers and not frequent_footers:
        return documents, {
            "enabled": True,
            "removed": False,
            "reason": "no_frequent_header_footer_detected",
            "threshold": threshold,
        }

    new_docs: List[Document] = []
    removed_any = False

    for doc in documents:
        txt = doc.get_content() or ""
        lines = (txt or "").splitlines()

        # Remove exact-line matches anywhere in the page, but only for frequent header/footer candidates.
        # This is conservative; you can strengthen later with positional removal.
        kept_lines: List[str] = []
        for ln in lines:
            s = ln.strip()
            if s and (s in frequent_headers or s in frequent_footers):
                removed_any = True
                continue
            kept_lines.append(ln)

        new_text = "\n".join(kept_lines).strip()
        new_doc = Document(text=new_text, metadata=dict(doc.metadata))
        new_docs.append(new_doc)

    return new_docs, {
        "enabled": True,
        "removed": removed_any,
        "threshold": threshold,
        "frequent_headers_count": len(frequent_headers),
        "frequent_footers_count": len(frequent_footers),
    }


# ----------------------------
# Chunking helpers
# ----------------------------
def _tokenize_paragraphs(text: str) -> List[str]:
    raw_blocks = re.split(r"\n\s*\n", text or "")
    return [block.strip() for block in raw_blocks if block.strip()]


def _chunk_deterministic(text: str, chunk_size: int, overlap: int) -> Iterable[Tuple[str, int, int]]:
    """
    Sliding character window (lightweight fallback).
    """
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = (text[start:end] or "").strip()
        if chunk:
            yield chunk, start, end
        if end >= n:
            break
        start = max(end - overlap, 0)


def _chunk_semantic_paragraph_aggregate(text: str, chunk_size: int) -> Iterable[Tuple[str, int, int]]:
    """
    Non-embedding “semantic-ish” fallback: aggregate by paragraphs.
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
            yield chunk_text, max(0, end_cursor - len(chunk_text)), end_cursor
            buffer = []
            current_len = 0

        buffer.append(para)
        current_len += para_len + 2
        cursor += para_len + 2

    if buffer:
        chunk_text = "\n\n".join(buffer).strip()
        yield chunk_text, max(0, cursor - len(chunk_text)), cursor


# ----------------------------
# PDF loading (P0/P1 observability)
# ----------------------------
def _load_pdf_documents(raw: bytes, pdf_path: str) -> Tuple[List[Document], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Return:
      - documents: List[Document] each representing a page
      - info: extraction metadata (method, etc.)
      - warnings: list of structured warning dicts

    Guarantees:
      - doc.metadata["page_index"] (0-based int)
      - doc.metadata["page_num"] (1-based int, derived from index, stable)
      - doc.metadata["page_label"] optional (display-only)
    """
    warnings: List[Dict[str, Any]] = []

    # 1) Prefer LlamaIndex + PyMuPDF
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

        if documents:
            # Enforce stable page_index based on order returned.
            fixed_docs: List[Document] = []
            for i, doc in enumerate(documents):
                md = dict(doc.metadata or {})
                page_label = md.get("page_label") or md.get("page_number")
                md["page_index"] = int(i)
                md["page_num"] = int(i + 1)
                if page_label is not None:
                    md["page_label"] = str(page_label)
                md["extraction_method"] = "llamaindex_pymupdf"
                fixed_docs.append(Document(text=doc.get_content() or "", metadata=md))

            return fixed_docs, {"extraction_method": "llamaindex_pymupdf"}, warnings

    except Exception as e:
        warnings.append(
            {
                "code": "PDF_EXTRACT_PYMUPDF_FAILED",
                "message": "Failed to extract PDF via LlamaIndex+PyMuPDFReader; falling back.",
                "details": {"error": repr(e)},
            }
        )
    finally:
        if tmp_path:
            _safe_unlink(tmp_path)

    # 2) Fallback: pypdf
    try:
        reader = PdfReader(io.BytesIO(raw))
        docs: List[Document] = []
        for idx, page in enumerate(reader.pages):
            extracted = page.extract_text() or ""
            md = {
                "page_index": idx,
                "page_num": idx + 1,
                "extraction_method": "pypdf",
            }
            docs.append(Document(text=extracted, metadata=md))
        if not docs:
            docs = [Document(text="", metadata={"page_index": 0, "page_num": 1, "extraction_method": "pypdf"})]
        return docs, {"extraction_method": "pypdf"}, warnings
    except Exception as e:
        warnings.append(
            {
                "code": "PDF_EXTRACT_PYPDF_FAILED",
                "message": "Failed to extract PDF via pypdf; final fallback is raw decode.",
                "details": {"error": repr(e)},
            }
        )

    # 3) Final fallback: decode bytes
    text = raw.decode("utf-8", errors="ignore")
    docs = [Document(text=text, metadata={"page_index": 0, "page_num": 1, "extraction_method": "bytes_decode"})]
    return docs, {"extraction_method": "bytes_decode"}, warnings


# ----------------------------
# Embedding model (P1/P3)
# ----------------------------
def _get_embed_model_with_reason() -> Tuple[Optional[Any], Optional[str]]:
    """
    Returns (embed_model, disabled_reason).
    disabled_reason is None if model is available.
    """
    global _embed_model

    if not _OPENAI_AVAILABLE:
        return None, "openai_embedding_import_unavailable"

    api_key = get_openai_api_key()
    if not api_key:
        return None, "missing_OPENAI_API_KEY"

    with _embed_lock:
        if _embed_model is not None:
            return _embed_model, None
        try:
            _embed_model = OpenAIEmbedding("text-embedding-3-small")
            return _embed_model, None
        except Exception as e:
            return None, f"openai_embedding_init_failed:{repr(e)}"


# ----------------------------
# IDs and meta (P0)
# ----------------------------
def _make_ids(*, doc_id: str, page_num: int, chunk_text: str) -> Tuple[str, str]:
    """
    Returns (chunk_id, content_id)
    - content_id: sha256(normalize(chunk_text))  (method-independent)
    - chunk_id: doc_id + page_num + content_id[:10] (page-sensitive, method-independent)
    """
    norm = _normalize_text_for_id(chunk_text)
    content_id = _sha256_str(norm)
    chunk_id = f"{doc_id}::p{page_num:04d}::c::{content_id[:10]}"
    return chunk_id, content_id


def _base_meta(
    *,
    doc_id: str,
    doc_hash: str,
    pdf_path: str,
    page_num: int,
    page_index: int,
    page_label: Optional[str],
    chunk_method: str,
    content_id: str,
    chunk_index_global: int,
    chunk_index_in_page: int,
    span_start: Optional[int],
    span_end: Optional[int],
    approx_tokens: int,
) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "doc_hash": doc_hash,
        "page": page_num,  # backward-compatible numeric page
        "page_num": page_num,
        "page_index": page_index,
        "page_label": page_label,
        "start": span_start,  # may be None if unknown
        "end": span_end,      # may be None if unknown
        "hash": content_id,   # content fingerprint (method-independent)
        "content_id": content_id,
        "chunk_method": chunk_method,
        "chunk_index": chunk_index_global,            # keep old key but now it is global
        "chunk_index_global": chunk_index_global,
        "chunk_index_in_page": chunk_index_in_page,
        "approx_tokens": approx_tokens,
        "source_path": os.path.abspath(pdf_path),
    }


# ----------------------------
# Convert to chunks (P0)
# ----------------------------
def _chunks_from_nodes(
    nodes: List[Any],
    *,
    doc_id: str,
    doc_hash: str,
    pdf_path: str,
    chunk_method: str,
    warnings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert LlamaIndex nodes to chunks.

    P0 change:
    - Do NOT pretend start/end offsets are known; set to None unless node provides explicit offsets.
    - Unify chunk indexing: global + per-page.
    """
    chunks: List[Dict[str, Any]] = []
    per_page_counter: Dict[int, int] = {}

    for global_idx, node in enumerate(nodes):
        chunk_text = node.get_content() or ""
        if not chunk_text.strip():
            continue

        md = dict(getattr(node, "metadata", {}) or {})
        # Prefer our metadata convention if present; fallback safely.
        page_index = md.get("page_index")
        page_num = md.get("page_num") or md.get("page")  # legacy
        page_label = md.get("page_label")

        try:
            if page_index is None:
                # If only page_num is present, derive index; else default to first page.
                if page_num is not None:
                    page_index = max(0, int(page_num) - 1)
                else:
                    page_index = 0
            page_index = int(page_index)
        except Exception:
            page_index = 0
            warnings.append(
                {"code": "NODE_PAGE_INDEX_INVALID", "message": "Invalid node page_index; defaulted to 0.", "details": {"node_meta": md}}
            )

        try:
            if page_num is None:
                page_num = page_index + 1
            page_num = int(page_num)
        except Exception:
            page_num = page_index + 1
            warnings.append(
                {"code": "NODE_PAGE_NUM_INVALID", "message": "Invalid node page_num/page; defaulted from page_index.", "details": {"node_meta": md}}
            )

        in_page = per_page_counter.get(page_index, 0)
        per_page_counter[page_index] = in_page + 1

        # Span offsets: only trust explicit values if present.
        span_start = md.get("start") or md.get("start_char") or md.get("start_char_idx")
        span_end = md.get("end") or md.get("end_char") or md.get("end_char_idx")
        try:
            span_start = int(span_start) if span_start is not None else None
        except Exception:
            span_start = None
        try:
            span_end = int(span_end) if span_end is not None else None
        except Exception:
            span_end = None

        chunk_id, content_id = _make_ids(doc_id=doc_id, page_num=page_num, chunk_text=chunk_text)
        meta = _base_meta(
            doc_id=doc_id,
            doc_hash=doc_hash,
            pdf_path=pdf_path,
            page_num=page_num,
            page_index=page_index,
            page_label=str(page_label) if page_label is not None else None,
            chunk_method=chunk_method,
            content_id=content_id,
            chunk_index_global=global_idx,
            chunk_index_in_page=in_page,
            span_start=span_start,
            span_end=span_end,
            approx_tokens=_approx_tokens(chunk_text),
        )

        chunks.append({"chunk_id": chunk_id, "text": chunk_text, "meta": meta})

    return chunks


def _chunks_from_custom_chunker(
    documents: List[Document],
    *,
    doc_id: str,
    doc_hash: str,
    pdf_path: str,
    chunk_method: str,
    chunker: Callable[..., Iterable[Tuple[str, int, int]]],
    chunker_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Custom chunker produces reliable (start,end) spans at page level.
    Provides unified global + per-page indexing.
    """
    chunks: List[Dict[str, Any]] = []
    global_idx = 0

    for doc in documents:
        md = dict(doc.metadata or {})
        page_index = int(md.get("page_index", 0))
        page_num = int(md.get("page_num", page_index + 1))
        page_label = md.get("page_label")
        page_text = doc.get_content() or ""

        in_page = 0
        for chunk_text, start, end in chunker(page_text, **chunker_kwargs):
            if not (chunk_text or "").strip():
                continue

            chunk_id, content_id = _make_ids(doc_id=doc_id, page_num=page_num, chunk_text=chunk_text)
            meta = _base_meta(
                doc_id=doc_id,
                doc_hash=doc_hash,
                pdf_path=pdf_path,
                page_num=page_num,
                page_index=page_index,
                page_label=str(page_label) if page_label is not None else None,
                chunk_method=chunk_method,
                content_id=content_id,
                chunk_index_global=global_idx,
                chunk_index_in_page=in_page,
                span_start=int(start),
                span_end=int(end),
                approx_tokens=_approx_tokens(chunk_text),
            )
            chunks.append({"chunk_id": chunk_id, "text": chunk_text, "meta": meta})
            global_idx += 1
            in_page += 1

    return chunks


# ----------------------------
# Chunk strategies
# ----------------------------
def _chunk_documents_deterministic(
    documents: List[Document],
    *,
    doc_id: str,
    doc_hash: str,
    pdf_path: str,
    chunk_size: int,
    overlap: int,
    warnings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Deterministic primary:
    - SentenceSplitter
    Fallback:
    - sliding window
    """
    try:
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        nodes = splitter.get_nodes_from_documents(documents)
        return _chunks_from_nodes(
            nodes,
            doc_id=doc_id,
            doc_hash=doc_hash,
            pdf_path=pdf_path,
            chunk_method="deterministic_sentence",
            warnings=warnings,
        )
    except Exception as e:
        warnings.append(
            {
                "code": "DETERMINISTIC_SENTENCE_SPLIT_FAILED",
                "message": "SentenceSplitter failed; falling back to sliding window chunking.",
                "details": {"error": repr(e)},
            }
        )
        return _chunks_from_custom_chunker(
            documents,
            doc_id=doc_id,
            doc_hash=doc_hash,
            pdf_path=pdf_path,
            chunk_method="deterministic_window",
            chunker=_chunk_deterministic,
            chunker_kwargs={"chunk_size": chunk_size, "overlap": overlap},
        )


def _chunk_documents_semantic(
    documents: List[Document],
    *,
    doc_id: str,
    doc_hash: str,
    pdf_path: str,
    chunk_size: int,
    overlap: int,
    warnings: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Semantic primary:
      1) SemanticSplitterNodeParser with embedding (if available)
      2) fallback SentenceSplitter
      3) fallback paragraph aggregation
      4) final fallback sliding window (overlap=0)

    Returns (chunks, semantic_info)
    """
    semantic_info: Dict[str, Any] = {"semantic_available": False, "semantic_disabled_reason": None}

    embed_model, disabled_reason = _get_embed_model_with_reason()
    if embed_model is None:
        semantic_info["semantic_available"] = False
        semantic_info["semantic_disabled_reason"] = disabled_reason
    else:
        semantic_info["semantic_available"] = True

        try:
            splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=embed_model,
            )
            nodes = splitter.get_nodes_from_documents(documents)
            chunks = _chunks_from_nodes(
                nodes,
                doc_id=doc_id,
                doc_hash=doc_hash,
                pdf_path=pdf_path,
                chunk_method="semantic_embedding",
                warnings=warnings,
            )
            return chunks, semantic_info
        except Exception as e:
            warnings.append(
                {
                    "code": "SEMANTIC_EMBEDDING_SPLIT_FAILED",
                    "message": "SemanticSplitterNodeParser failed; falling back.",
                    "details": {"error": repr(e)},
                }
            )

    # fallback 1: SentenceSplitter
    try:
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        nodes = splitter.get_nodes_from_documents(documents)
        chunks = _chunks_from_nodes(
            nodes,
            doc_id=doc_id,
            doc_hash=doc_hash,
            pdf_path=pdf_path,
            chunk_method="semantic_fallback_sentence",
            warnings=warnings,
        )
        return chunks, semantic_info
    except Exception as e:
        warnings.append(
            {
                "code": "SEMANTIC_FALLBACK_SENTENCE_FAILED",
                "message": "SentenceSplitter fallback failed; falling back to paragraph aggregation.",
                "details": {"error": repr(e)},
            }
        )

    # fallback 2: paragraph aggregation
    try:
        chunks = _chunks_from_custom_chunker(
            documents,
            doc_id=doc_id,
            doc_hash=doc_hash,
            pdf_path=pdf_path,
            chunk_method="semantic_fallback_paragraph",
            chunker=_chunk_semantic_paragraph_aggregate,
            chunker_kwargs={"chunk_size": chunk_size},
        )
        return chunks, semantic_info
    except Exception as e:
        warnings.append(
            {
                "code": "SEMANTIC_FALLBACK_PARAGRAPH_FAILED",
                "message": "Paragraph aggregation fallback failed; final fallback is sliding window.",
                "details": {"error": repr(e)},
            }
        )

    # final fallback: sliding window no overlap
    chunks = _chunks_from_custom_chunker(
        documents,
        doc_id=doc_id,
        doc_hash=doc_hash,
        pdf_path=pdf_path,
        chunk_method="semantic_fallback_window",
        chunker=_chunk_deterministic,
        chunker_kwargs={"chunk_size": chunk_size, "overlap": 0},
    )
    return chunks, semantic_info


def _should_use_semantic(
    *,
    raw_size_bytes: int,
    page_count: int,
    prefer_deterministic: bool,
    importance: str,
    semantic_max_bytes: int,
    semantic_max_pages: int,
) -> bool:
    if prefer_deterministic:
        return False
    if str(importance).lower() == "low":
        return False
    if raw_size_bytes > semantic_max_bytes:
        return False
    if page_count > semantic_max_pages:
        return False
    return True


# ----------------------------
# Tool entry
# ----------------------------

def pdf_ingest_tool(args: Dict[str, Any]) -> ToolResult:
    """
    args:
      - pdf_path: str
      - chunk_size: int (optional)
      - overlap: int (optional)
      - segmentation: "semantic" | "deterministic" (optional)
      - doc_id: str (optional)

    strategy knobs:
      - prefer_deterministic: bool (default False)
      - importance: "high"|"normal"|"low" (default "normal")
      - semantic_max_bytes: int (default 2_000_000)
      - semantic_max_pages: int (default 200)

    cleaning knobs (P2):
      - clean_text: bool (default True)
      - merge_hyphenation: bool (default True)
      - collapse_blank_lines: bool (default True)
      - remove_headers_footers: bool (default False)
      - header_lines: int (default 2)
      - footer_lines: int (default 2)
      - header_footer_min_repetition_ratio: float (default 0.35)

    returns:
      - chunks: [{chunk_id, text, meta}]
      - plus observability fields: warnings, extraction_method, semantic_available, semantic_disabled_reason
    """
    tool_name = "pdf_ingest_tool"
    warnings: List[Dict[str, Any]] = []
    timing_ms: Dict[str, int] = {}

    def _fail(code: str, message: str, details: Optional[Dict[str, Any]] = None) -> ToolResult:
        return ToolResult(
            ok=False,
            tool_name=tool_name,
            args=normalized_args,
            error=ToolError(code=code, message=message, details=details),
            meta={
                "warnings": warnings,
                "timing_ms": timing_ms,
                "tool_version": "2025-12-22",
                "extraction_method": (extract_info or {}).get("extraction_method") if "extract_info" in locals() else None,
            },
        )

    # ---- validate ----
    if "pdf_path" not in args:
        return _fail("IO_ERROR", "Failed to read PDF", {"pdf_path": pdf_path, "error": repr(e)})


    pdf_path = str(args["pdf_path"])
    chunk_size = max(64, int(args.get("chunk_size") or _DEFAULT_CHUNK_SIZE))
    overlap = max(0, int(args.get("overlap") or _DEFAULT_OVERLAP))
    segmentation = str(args.get("segmentation") or "deterministic").lower()
    doc_id = args.get("doc_id")

    if segmentation not in {"semantic", "deterministic"}:
        return _fail("VALIDATION_ERROR", "method not in semantic/deterministic", {"pdf_path": pdf_path, "error": repr(e)})

    # ---- strategy knobs ----
    prefer_deterministic = bool(args.get("prefer_deterministic") or True)
    importance = str(args.get("importance") or "normal")
    semantic_max_bytes = int(args.get("semantic_max_bytes") or 2_000_000)
    semantic_max_pages = int(args.get("semantic_max_pages") or 200)

    # ---- cleaning knobs ----
    clean_text_enabled = bool(args.get("clean_text", True))
    merge_hyphenation = bool(args.get("merge_hyphenation", True))
    collapse_blank_lines = bool(args.get("collapse_blank_lines", True))

    remove_hf = bool(args.get("remove_headers_footers", False))
    header_lines = int(args.get("header_lines", 2))
    footer_lines = int(args.get("footer_lines", 2))
    hf_ratio = float(args.get("header_footer_min_repetition_ratio", 0.35))

    normalized_args = {
        "pdf_path": pdf_path,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "segmentation": segmentation,
        "doc_id": doc_id,
        "prefer_deterministic": prefer_deterministic,
        "importance": importance,
        "semantic_max_bytes": semantic_max_bytes,
        "semantic_max_pages": semantic_max_pages,
        "clean_text": clean_text_enabled,
        "merge_hyphenation": merge_hyphenation,
        "collapse_blank_lines": collapse_blank_lines,
        "remove_headers_footers": remove_hf,
        "header_lines": header_lines,
        "footer_lines": footer_lines,
        "header_footer_min_repetition_ratio": hf_ratio,
    }

    # ---- read file ----
    try:
        with open(pdf_path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return _fail("FILE_NOT_FOUND", "File not found", {"pdf_path": pdf_path, "error": repr(e)})
    except Exception as e:
        return _fail("IO_ERROR", "Failed to read PDF", {"pdf_path": pdf_path, "error": repr(e)})


    doc_hash = _sha256_bytes(raw)
    inferred_doc_id = str(doc_id) if doc_id else f"doc::{doc_hash[:12]}"

    # ---- extraction ----
    warnings: List[Dict[str, Any]] = []
    documents, extract_info, extract_warnings = _load_pdf_documents(raw, pdf_path)
    warnings.extend(extract_warnings)

    if not documents:
        return _fail("IO_EPROCESSING_ERROR", "not documents", {"pdf_path": pdf_path, "error": repr(e)})

    page_count = len(documents)

    # ---- optional cleaning ----
    cleaning_info: Dict[str, Any] = {"clean_text": clean_text_enabled, "remove_headers_footers": remove_hf}
    if clean_text_enabled:
        cleaned_docs: List[Document] = []
        for doc in documents:
            md = dict(doc.metadata or {})
            txt = doc.get_content() or ""
            txt = _clean_text(txt, merge_hyphenation=merge_hyphenation, collapse_blank_lines=collapse_blank_lines)
            cleaned_docs.append(Document(text=txt, metadata=md))
        documents = cleaned_docs
        cleaning_info.update({"merge_hyphenation": merge_hyphenation, "collapse_blank_lines": collapse_blank_lines})

    if remove_hf:
        documents, hf_info = _remove_headers_footers(
            documents,
            header_lines=header_lines,
            footer_lines=footer_lines,
            min_repetition_ratio=hf_ratio,
        )
        cleaning_info["header_footer"] = hf_info

    # ---- decide segmentation ----
    use_semantic = segmentation == "semantic" and _should_use_semantic(
        raw_size_bytes=len(raw),
        page_count=page_count,
        prefer_deterministic=prefer_deterministic,
        importance=importance,
        semantic_max_bytes=semantic_max_bytes,
        semantic_max_pages=semantic_max_pages,
    )

    semantic_info: Dict[str, Any] = {"semantic_available": False, "semantic_disabled_reason": None}

    if use_semantic:
        chunks, semantic_info = _chunk_documents_semantic(
            documents,
            doc_id=inferred_doc_id,
            doc_hash=doc_hash,
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            overlap=overlap,
            warnings=warnings,
        )
        chosen = "semantic"
    else:
        # If user asked semantic but we downgraded, record reason
        if segmentation == "semantic":
            semantic_info["semantic_available"] = False
            semantic_info["semantic_disabled_reason"] = {
                "prefer_deterministic": prefer_deterministic,
                "importance": importance,
                "raw_size_bytes": len(raw),
                "page_count": page_count,
                "semantic_max_bytes": semantic_max_bytes,
                "semantic_max_pages": semantic_max_pages,
            }
        chunks = _chunk_documents_deterministic(
            documents,
            doc_id=inferred_doc_id,
            doc_hash=doc_hash,
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            overlap=overlap,
            warnings=warnings,
        )
        chosen = "deterministic"

    meta = {
        "warnings": warnings,
        "extraction_method": extract_info.get("extraction_method"),
        "chosen_segmentation": chosen,
        "timing_ms": timing_ms,      # 可选
        "tool_version": "2025-12-22",  # 可选
        "semantic_available": semantic_info.get("semantic_available"),
        "semantic_disabled_reason": semantic_info.get("semantic_disabled_reason"),
        "cleaning": cleaning_info,
    }

    return ToolResult(
        ok=True,
        tool_name=tool_name,
        data={
            "doc_id": inferred_doc_id,
            "doc_hash": doc_hash,
            "page_count": page_count,
            "chunks": chunks,
        },
        args=normalized_args,
        meta=meta,
    )
