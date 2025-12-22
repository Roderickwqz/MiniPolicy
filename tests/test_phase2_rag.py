#!/usr/bin/env python3
"""
Test script for Phase 2 RAG implementation.
Tests stable chunk_id generation and Weaviate integration.
"""

import os
import sys
import json
from pathlib import Path
import dotenv

# Add project root to path so we can import app module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from app.mcp.tools.pdf_ingest import pdf_ingest_tool
from app.mcp.tools.vector_index import vector_index_tool
from app.mcp.tools.semantic_retrieve import semantic_retrieve_tool
dotenv.load_dotenv()

def test_stable_chunk_id():
    """Test that same PDF re-run produces identical chunk_ids."""
    print("Testing stable chunk_id generation...")
    
    # Use a sample PDF (you'll need to provide one)
    pdf_path = os.getenv("TEST_PDF_PATH", "data/gdpr-google.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"  ⚠️  Skipping: PDF not found at {pdf_path}")
        print("     Set TEST_PDF_PATH environment variable to test")
        return False
    
    # First run
    args1 = {
        "pdf_path": pdf_path,
        "chunk_size": 800,
        "overlap": 120,
        "segmentation": "deterministic",
    }
    result1 = pdf_ingest_tool(args1)
    
    if not result1.ok:
        print(f"  ❌ First ingest failed: {result1.error}")
        return False
    
    chunks1 = result1.data.get("chunks", [])
    chunk_ids1 = [chunk["chunk_id"] for chunk in chunks1]
    
    # Second run (same parameters)
    result2 = pdf_ingest_tool(args1)
    
    if not result2.ok:
        print(f"  ❌ Second ingest failed: {result2.error}")
        return False
    
    chunks2 = result2.data.get("chunks", [])
    chunk_ids2 = [chunk["chunk_id"] for chunk in chunks2]
    
    # Compare
    if chunk_ids1 == chunk_ids2:
        print(f"  ✅ PASS: {len(chunk_ids1)} chunk_ids are stable")
        return True
    else:
        print(f"  ❌ FAIL: Chunk IDs differ between runs")
        print(f"     First run: {len(chunk_ids1)} chunks")
        print(f"     Second run: {len(chunk_ids2)} chunks")
        if len(chunk_ids1) == len(chunk_ids2):
            for i, (id1, id2) in enumerate(zip(chunk_ids1, chunk_ids2)):
                if id1 != id2:
                    print(f"     First difference at index {i}: {id1} != {id2}")
                    break
        return False


def _check_weaviate_connection():
    """Helper: Check if Weaviate is available."""
    try:
        from app.mcp.tools.weaviate_client import get_weaviate_client
        client = get_weaviate_client()
        if client is None:
            return False, "Weaviate client not available. Make sure Weaviate is running (docker-compose up)"
        return True, None
    except Exception as e:
        return False, f"Cannot connect to Weaviate: {e}"


def _get_test_pdf_path():
    """Helper: Get test PDF path."""
    pdf_path = os.getenv("TEST_PDF_PATH", "data/gdpr-google.pdf")
    if not os.path.exists(pdf_path):
        return None, f"PDF not found at {pdf_path}"
    return pdf_path, None


def test_pdf_ingest(method='deterministic'):
    """Test PDF ingestion step."""
    print("\nTesting PDF Ingest...")
    
    pdf_path, error = _get_test_pdf_path()
    if error:
        print(f"  ⚠️  Skipping: {error}")
        return False
    
    ingest_args = {
        "pdf_path": pdf_path,
        "chunk_size": 800,
        "overlap": 120,
        "segmentation": method,  # semantic | deterministic
    }
    ingest_result = pdf_ingest_tool(ingest_args)
    
    if not ingest_result.ok:
        print(f"  ❌ Ingest failed: {ingest_result.error}")
        return False
    
    chunks = ingest_result.data.get("chunks", [])
    if len(chunks) == 0:
        print(f"  ❌ No chunks extracted")
        return False
    
    # Verify chunk structure
    required_fields = ["chunk_id", "text", "meta"]
    all_valid = True
    for chunk in chunks[:5]:  # Check first 5 chunks
        for field in required_fields:
            if field not in chunk:
                print(f"  ❌ Chunk missing {field}: {chunk}")
                all_valid = False
    
    if not all_valid:
        return False
    
    print(f"  ✅ Ingested {len(chunks)} chunks")
    print(f"     Chosen segmentation: {ingest_result.meta.get('chosen_segmentation', 'unknown')}")
    print(f"     Page count: {ingest_result.data.get('page_count', 0)}")
    return True


def test_vector_index():
    """Test vector indexing step."""
    print("\nTesting Vector Index (Weaviate)...")
    
    # Check Weaviate connection
    connected, error = _check_weaviate_connection()
    if not connected:
        print(f"  ⚠️  Skipping: {error}")
        return False
    
    # Get PDF and ingest first
    pdf_path, error = _get_test_pdf_path()
    if error:
        print(f"  ⚠️  Skipping: {error}")
        return False
    
    ingest_args = {
        "pdf_path": pdf_path,
        "chunk_size": 800,
        "overlap": 120,
        "segmentation": "deterministic",
    }
    ingest_result = pdf_ingest_tool(ingest_args)
    
    if not ingest_result.ok:
        print(f"  ❌ Ingest failed: {ingest_result.error}")
        return False
    
    chunks = ingest_result.data.get("chunks", [])
    if len(chunks) == 0:
        print(f"  ❌ No chunks to index")
        return False
    
    # Test indexing
    index_name = "TestPhase2Index"
    index_args = {
        "index_name": index_name,
        "chunks": chunks,
    }
    index_result = vector_index_tool(index_args)
    
    if not index_result.ok:
        print(f"  ❌ Indexing failed: {index_result.error}")
        return False
    
    upserted = index_result.data.get("upserted", 0)
    index_size = index_result.data.get("index_size", 0)
    
    if upserted == 0:
        print(f"  ❌ No chunks were indexed")
        return False
    
    print(f"  ✅ Indexed {upserted} chunks")
    print(f"     Index size: {index_size}")
    print(f"     Class name: {index_result.data.get('class_name', 'unknown')}")
    return True


def test_semantic_retrieve():
    """Test semantic retrieval step."""
    print("\nTesting Semantic Retrieve...")
    
    # Check Weaviate connection
    connected, error = _check_weaviate_connection()
    if not connected:
        print(f"  ⚠️  Skipping: {error}")
        return False
    
    # Get PDF, ingest, and index first
    pdf_path, error = _get_test_pdf_path()
    if error:
        print(f"  ⚠️  Skipping: {error}")
        return False
    
    ingest_args = {
        "pdf_path": pdf_path,
        "chunk_size": 800,
        "overlap": 120,
        "segmentation": "deterministic",
    }
    ingest_result = pdf_ingest_tool(ingest_args)
    
    if not ingest_result.ok:
        print(f"  ❌ Ingest failed: {ingest_result.error}")
        return False
    
    chunks = ingest_result.data.get("chunks", [])
    if len(chunks) == 0:
        print(f"  ❌ No chunks to index")
        return False
    
    index_name = "TestPhase2Index"
    index_args = {
        "index_name": index_name,
        "chunks": chunks,
    }
    index_result = vector_index_tool(index_args)
    
    if not index_result.ok:
        print(f"  ❌ Indexing failed: {index_result.error}")
        return False
    
    # Test retrieval
    query = "What are the main requirements?"
    retrieve_args = {
        "index_name": index_name,
        "query": query,
        "top_k": 3,
    }
    retrieve_result = semantic_retrieve_tool(retrieve_args)
    
    if not retrieve_result.ok:
        print(f"  ❌ Retrieval failed: {retrieve_result.error}")
        return False
    
    matches = retrieve_result.data.get("matches", [])
    if len(matches) == 0:
        print(f"  ⚠️  No matches found (might be expected for some queries)")
        return True  # Not necessarily a failure
    
    print(f"  ✅ Retrieved {len(matches)} matches")
    print(f"     Query: {query}")
    if matches:
        print(f"     Top match score: {matches[0].get('score', 0):.4f}")
    return True


def test_skill_envelope_format():
    """Test Skill Envelope format validation."""
    print("\nTesting Skill Envelope format...")
    
    # Check Weaviate connection
    connected, error = _check_weaviate_connection()
    if not connected:
        print(f"  ⚠️  Skipping: {error}")
        return False
    
    # Get PDF, ingest, index, and retrieve
    pdf_path, error = _get_test_pdf_path()
    if error:
        print(f"  ⚠️  Skipping: {error}")
        return False
    
    ingest_args = {
        "pdf_path": pdf_path,
        "chunk_size": 800,
        "overlap": 120,
        "segmentation": "deterministic",
    }
    ingest_result = pdf_ingest_tool(ingest_args)
    
    if not ingest_result.ok:
        print(f"  ❌ Ingest failed: {ingest_result.error}")
        return False
    
    chunks = ingest_result.data.get("chunks", [])
    if len(chunks) == 0:
        print(f"  ❌ No chunks to index")
        return False
    
    index_name = "TestPhase2Index"
    index_args = {
        "index_name": index_name,
        "chunks": chunks,
    }
    index_result = vector_index_tool(index_args)
    
    if not index_result.ok:
        print(f"  ❌ Indexing failed: {index_result.error}")
        return False
    
    query = "What are the main requirements?"
    retrieve_args = {
        "index_name": index_name,
        "query": query,
        "top_k": 3,
    }
    retrieve_result = semantic_retrieve_tool(retrieve_args)
    
    if not retrieve_result.ok:
        print(f"  ❌ Retrieval failed: {retrieve_result.error}")
        return False
    
    matches = retrieve_result.data.get("matches", [])
    if len(matches) == 0:
        print(f"  ⚠️  No matches to validate format")
        return True  # Not necessarily a failure
    
    # Validate format
    required_fields = ["chunk_id", "text", "score", "meta"]
    all_valid = True
    for match in matches:
        for field in required_fields:
            if field not in match:
                print(f"  ❌ Match missing {field}: {match}")
                all_valid = False
    
    if not all_valid:
        return False
    
    # Verify evidence chain (chunk_id present)
    for match in matches:
        meta = match.get("meta", {})
        chunk_id = match.get("chunk_id")
        if not chunk_id:
            print(f"  ❌ Match missing chunk_id at top level: {match}")
            all_valid = False
    
    if not all_valid:
        return False
    
    print(f"  ✅ All {len(matches)} matches have required fields (chunk_id, text, score, meta)")
    print(f"  ✅ Evidence chain verified (chunk_id present)")
    return True


def test_no_results_handling():
    """Test that 'no results' case is handled properly."""
    print("\nTesting 'no results' handling...")
    
    # Try to retrieve from a non-existent index or with a query that won't match
    index_name = "nonexistent_index"
    query = "xyzabc123nonexistentquery"
    
    retrieve_args = {
        "index_name": index_name,
        "query": query,
        "top_k": 3,
    }
    retrieve_result = semantic_retrieve_tool(retrieve_args)
    
    # Should either return empty matches or an error
    if retrieve_result.ok:
        matches = retrieve_result.data.get("matches", [])
        if len(matches) == 0:
            print("  ✅ No results case handled (empty matches)")
            return True
        else:
            print(f"  ⚠️  Got {len(matches)} matches for nonexistent query")
            return False
    else:
        # Error is also acceptable for nonexistent index
        print(f"  ✅ No results case handled (error returned: {retrieve_result.error})")
        return True


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2 RAG Implementation Tests")
    print("=" * 60)
    
    results = []
    
    # Test 1: Stable chunk_id
    # results.append(("Stable chunk_id", test_stable_chunk_id()))
    
    # Test 2: PDF Ingest (Step 1)
    results.append(("PDF Ingest", test_pdf_ingest(method='semantic')))
    
    # Test 3: Vector Index (Step 2)
    # results.append(("Vector Index", test_vector_index()))
    
    # Test 4: Semantic Retrieve (Step 3)
    # results.append(("Semantic Retrieve", test_semantic_retrieve()))
    
    # Test 5: Skill Envelope Format (Step 4)
    # results.append(("Skill Envelope Format", test_skill_envelope_format()))
    
    # Test 6: No results handling
    # results.append(("No results handling", test_no_results_handling()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL / ⚠️  SKIP"
        print(f"  {name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    sys.exit(0 if all_passed else 1)
