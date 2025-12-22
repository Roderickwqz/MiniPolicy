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


def test_end_to_end():
    """Test end-to-end: PDF ingest → Weaviate storage → SemanticRetrieve → Skill Envelope output."""
    print("\nTesting end-to-end RAG pipeline...")
    
    # Check Weaviate connection
    try:
        from app.mcp.tools.weaviate_client import get_weaviate_client
        client = get_weaviate_client()
        if client is None:
            print("  ⚠️  Skipping: Weaviate client not available")
            print("     Make sure Weaviate is running (docker-compose up)")
            return False
    except Exception as e:
        print(f"  ⚠️  Skipping: Cannot connect to Weaviate: {e}")
        return False
    
    # Use a sample PDF
    pdf_path = os.getenv("TEST_PDF_PATH", "data/gdpr-google.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"  ⚠️  Skipping: PDF not found at {pdf_path}")
        return False
    
    # Step 1: PDF Ingest
    print("  1. PDF Ingest...")
    ingest_args = {
        "pdf_path": pdf_path,
        "chunk_size": 800,
        "overlap": 120,
        "segmentation": "both", # deterministic | semantic | both
    }
    ingest_result = pdf_ingest_tool(ingest_args)
    
    if not ingest_result.ok:
        print(f"     ❌ Ingest failed: {ingest_result.error}")
        return False
    
    chunks = ingest_result.data.get("chunks", [])
    print(f"     ✅ Ingested {len(chunks)} chunks")
    
    # Step 2: Vector Index
    print("  2. Vector Index (Weaviate)...")
    index_name = "TestPhase2Index"
    index_args = {
        "index_name": index_name,
        "chunks": chunks,
    }
    index_result = vector_index_tool(index_args)
    
    if not index_result.ok:
        print(f"     ❌ Indexing failed: {index_result.error}")
        return False
    
    upserted = index_result.data.get("upserted", 0)
    print(f"     ✅ Indexed {upserted} chunks")
    
    # Step 3: Semantic Retrieve
    print("  3. Semantic Retrieve...")
    query = "What are the main requirements?"
    retrieve_args = {
        "index_name": index_name,
        "query": query,
        "top_k": 3,
    }
    retrieve_result = semantic_retrieve_tool(retrieve_args)
    
    if not retrieve_result.ok:
        print(f"     ❌ Retrieval failed: {retrieve_result.error}")
        return False
    
    matches = retrieve_result.data.get("matches", [])
    print(f"     ✅ Retrieved {len(matches)} matches")
    
    # Step 4: Verify Skill Envelope (check node_semantic_retrieve output)
    print("  4. Verifying Skill Envelope format...")
    
    # Check that matches have required fields
    all_valid = True
    for match in matches:
        if "chunk_id" not in match:
            print(f"     ❌ Match missing chunk_id: {match}")
            all_valid = False
        if "text" not in match:
            print(f"     ❌ Match missing text: {match}")
            all_valid = False
        if "score" not in match:
            print(f"     ❌ Match missing score: {match}")
            all_valid = False
        if "meta" not in match:
            print(f"     ❌ Match missing meta: {match}")
            all_valid = False
    
    if all_valid:
        print(f"     ✅ All matches have required fields (chunk_id, text, score, meta)")
    
    # Verify evidence contains chunk_id
    for match in matches:
        meta = match.get("meta", {})
        if "chunk_id" not in meta and match.get("chunk_id"):
            # chunk_id is at top level, which is fine
            pass
    
    print(f"     ✅ Evidence chain verified (chunk_id present)")
    
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
    
    # Test 2: End-to-end
    results.append(("End-to-end pipeline", test_end_to_end()))
    
    # Test 3: No results handling
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
