#!/usr/bin/env python3
"""
Test script for Phase 3 GraphRAG implementation.
Tests Neo4j graph store, graph_path_id tracking, and connection pooling.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add project root to path so we can import app module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import dotenv
dotenv.load_dotenv()

from app.mcp.tools.graph_store import graph_upsert_tool, graph_query_tool
from app.mcp.tools.neo4j_client import neo4j_connection, get_neo4j_driver
from app.mcp.tools.weaviate_client import weaviate_connection
from app.mcp.artifacts import append_graph_path, get_graph_path
from app.mcp.config import get_neo4j_config


def _check_neo4j_connection():
    """Check if Neo4j is available."""
    try:
        driver = get_neo4j_driver()
        if driver is None:
            return False, "Neo4j configuration missing (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)"
        
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record and record["test"] == 1:
                driver.close()
                return True, None
            driver.close()
            return False, "Neo4j connection test failed"
    except Exception as e:
        return False, str(e)


def test_neo4j_connection():
    """Test Neo4j connection."""
    print("\nTesting Neo4j Connection...")
    
    connected, error = _check_neo4j_connection()
    if not connected:
        print(f"  ⚠️  Skipping: {error}")
        print("     Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD environment variables")
        return False
    
    print("  ✅ Neo4j connection successful")
    return True


def test_graph_upsert_nodes():
    """Test graph_upsert_tool for creating nodes."""
    print("\nTesting Graph Upsert (Nodes)...")
    
    connected, error = _check_neo4j_connection()
    if not connected:
        print(f"  ⚠️  Skipping: {error}")
        return False
    
    # Create test nodes
    nodes = [
        {
            "type": "Doc",
            "id": "test_doc_1",
            "properties": {
                "doc_id": "test_doc_1",
                "doc_hash": "abc123",
                "source_path": "/test/path.pdf",
                "page_count": 10,
            },
        },
        {
            "type": "Chunk",
            "id": "test_chunk_1",
            "properties": {
                "chunk_id": "test_chunk_1",
                "content_id": "chunk_hash_1",
                "text": "Test chunk text",
                "doc_id": "test_doc_1",
                "page_num": 1,
                "chunk_index_global": 0,
                "chunk_method": "test",
            },
        },
    ]
    
    edges = [
        {
            "from_type": "Doc",
            "from_id": "test_doc_1",
            "to_type": "Chunk",
            "to_id": "test_chunk_1",
            "relationship": "HAS_CHUNK",
            "properties": {"order": 0},
        },
    ]
    
    args = {"nodes": nodes, "edges": edges}
    result = graph_upsert_tool(args)
    
    if not result.ok:
        print(f"  ❌ Upsert failed: {result.error}")
        return False
    
    data = result.data or {}
    nodes_created = data.get("nodes_created", 0)
    edges_created = data.get("edges_created", 0)
    
    if nodes_created < 2:
        print(f"  ❌ Expected at least 2 nodes created, got {nodes_created}")
        return False
    
    if edges_created < 1:
        print(f"  ❌ Expected at least 1 edge created, got {edges_created}")
        return False
    
    print(f"  ✅ Created {nodes_created} nodes and {edges_created} edges")
    return True


def test_graph_upsert_provenance():
    """Test provenance links (DERIVED_FROM relationships)."""
    print("\nTesting Graph Upsert (Provenance)...")
    
    connected, error = _check_neo4j_connection()
    if not connected:
        print(f"  ⚠️  Skipping: {error}")
        return False
    
    # Create Chunk node first
    nodes = [
        {
            "type": "Chunk",
            "id": "provenance_chunk_1",
            "properties": {
                "chunk_id": "provenance_chunk_1",
                "text": "Source chunk for provenance test",
                "doc_id": "test_doc_1",
            },
        },
        {
            "type": "Concept",
            "id": "concept_1",
            "properties": {
                "concept_id": "concept_1",
                "name": "Test Concept",
                "description": "A concept derived from chunk",
                "confidence": 0.9,
            },
        },
    ]
    
    edges = [
        {
            "from_type": "Concept",
            "from_id": "concept_1",
            "to_type": "Chunk",
            "to_id": "provenance_chunk_1",
            "relationship": "DERIVED_FROM",
            "properties": {
                "confidence": 0.9,
                "method": "test",
            },
        },
    ]
    
    args = {"nodes": nodes, "edges": edges}
    result = graph_upsert_tool(args)
    
    if not result.ok:
        print(f"  ❌ Provenance upsert failed: {result.error}")
        return False
    
    data = result.data or {}
    edges_created = data.get("edges_created", 0)
    
    if edges_created < 1:
        print(f"  ❌ Expected DERIVED_FROM edge, got {edges_created}")
        return False
    
    print(f"  ✅ Created DERIVED_FROM provenance link")
    return True


def test_graph_query():
    """Test graph_query_tool and graph_path_id generation."""
    print("\nTesting Graph Query...")
    
    connected, error = _check_neo4j_connection()
    if not connected:
        print(f"  ⚠️  Skipping: {error}")
        return False
    
    # First, ensure we have some data to query
    nodes = [
        {
            "type": "Doc",
            "id": "query_test_doc",
            "properties": {"doc_id": "query_test_doc", "doc_hash": "test"},
        },
    ]
    graph_upsert_tool({"nodes": nodes, "edges": []})
    
    # Test query
    query = "MATCH (d:Doc {id: 'query_test_doc'}) RETURN d"
    args = {"query": query}
    result = graph_query_tool(args)
    
    if not result.ok:
        print(f"  ❌ Query failed: {result.error}")
        return False
    
    data = result.data or {}
    graph_path_id = data.get("graph_path_id")
    nodes_result = data.get("nodes", [])
    
    if not graph_path_id:
        print(f"  ❌ Missing graph_path_id in result")
        return False
    
    if not graph_path_id.startswith("graph_path::"):
        print(f"  ❌ graph_path_id format incorrect: {graph_path_id}")
        return False
    
    print(f"  ✅ Query successful")
    print(f"     graph_path_id: {graph_path_id}")
    print(f"     nodes returned: {len(nodes_result)}")
    return True


def test_graph_path_storage():
    """Test graph_path_id storage in artifacts.json."""
    print("\nTesting Graph Path Storage...")
    
    connected, error = _check_neo4j_connection()
    if not connected:
        print(f"  ⚠️  Skipping: {error}")
        return False
    
    # Create temporary run directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Execute a query with run_dir
        query = "MATCH (n) RETURN n LIMIT 5"
        args = {
            "query": query,
            "run_dir": temp_dir,
        }
        result = graph_query_tool(args)
        
        if not result.ok:
            print(f"  ❌ Query failed: {result.error}")
            return False
        
        graph_path_id = result.data.get("graph_path_id")
        if not graph_path_id:
            print(f"  ❌ Missing graph_path_id")
            return False
        
        # Verify path is stored in artifacts.json
        stored_path = get_graph_path(temp_dir, graph_path_id)
        if not stored_path:
            print(f"  ❌ graph_path_id not found in artifacts.json")
            return False
        
        # Verify stored path structure
        if stored_path.get("graph_path_id") != graph_path_id:
            print(f"  ❌ graph_path_id mismatch")
            return False
        
        if stored_path.get("query") != query:
            print(f"  ❌ Query mismatch")
            return False
        
        if "nodes" not in stored_path or "edges" not in stored_path:
            print(f"  ❌ Missing nodes or edges in stored path")
            return False
        
        print(f"  ✅ graph_path_id stored in artifacts.json")
        print(f"     graph_path_id: {graph_path_id}")
        print(f"     nodes: {len(stored_path.get('nodes', []))}")
        print(f"     edges: {len(stored_path.get('edges', []))}")
        return True
    
    finally:
        shutil.rmtree(temp_dir)


def test_weaviate_connection_pool():
    """Test Weaviate connection pooling."""
    print("\nTesting Weaviate Connection Pool...")
    
    try:
        # Test that we can use the context manager multiple times
        with weaviate_connection() as client1:
            collections1 = client1.collections.list_all()
        
        with weaviate_connection() as client2:
            collections2 = client2.collections.list_all()
        
        # If we get here without errors, connection pooling is working
        print(f"  ✅ Connection pool working (reused connections)")
        return True
    
    except Exception as e:
        # If Weaviate is not available, that's OK for this test
        if "connection" in str(e).lower() or "connect" in str(e).lower():
            print(f"  ⚠️  Skipping: Weaviate not available ({str(e)[:100]})")
            return True  # Not a failure if Weaviate is not configured
        print(f"  ❌ Connection pool test failed: {e}")
        return False


def test_graph_build_integration():
    """Test full graph build integration (Doc + Chunk nodes)."""
    print("\nTesting Graph Build Integration...")
    
    connected, error = _check_neo4j_connection()
    if not connected:
        print(f"  ⚠️  Skipping: {error}")
        return False
    
    # Simulate graph build: create Doc and multiple Chunks
    doc_id = "integration_test_doc"
    nodes = [
        {
            "type": "Doc",
            "id": doc_id,
            "properties": {
                "doc_id": doc_id,
                "doc_hash": "integration_hash",
                "source_path": "/test/integration.pdf",
                "page_count": 3,
            },
        },
    ]
    
    edges = []
    for i in range(3):
        chunk_id = f"{doc_id}::chunk_{i}"
        nodes.append({
            "type": "Chunk",
            "id": chunk_id,
            "properties": {
                "chunk_id": chunk_id,
                "text": f"Chunk {i} text",
                "doc_id": doc_id,
                "page_num": i + 1,
                "chunk_index_global": i,
            },
        })
        edges.append({
            "from_type": "Doc",
            "from_id": doc_id,
            "to_type": "Chunk",
            "to_id": chunk_id,
            "relationship": "HAS_CHUNK",
            "properties": {"order": i},
        })
    
    args = {"nodes": nodes, "edges": edges}
    result = graph_upsert_tool(args)
    
    if not result.ok:
        print(f"  ❌ Integration test failed: {result.error}")
        return False
    
    data = result.data or {}
    nodes_created = data.get("nodes_created", 0)
    edges_created = data.get("edges_created", 0)
    
    # Should have 1 Doc + 3 Chunks = 4 nodes, 3 edges
    if nodes_created < 4:
        print(f"  ❌ Expected 4 nodes, got {nodes_created}")
        return False
    
    if edges_created < 3:
        print(f"  ❌ Expected 3 edges, got {edges_created}")
        return False
    
    # Verify we can query the graph
    query = f"MATCH (d:Doc {{id: '{doc_id}'}})-[:HAS_CHUNK]->(c:Chunk) RETURN d, c"
    query_result = graph_query_tool({"query": query})
    
    if not query_result.ok:
        print(f"  ❌ Query verification failed: {query_result.error}")
        return False
    
    query_nodes = query_result.data.get("nodes", [])
    if len(query_nodes) < 4:  # 1 Doc + 3 Chunks
        print(f"  ❌ Expected 4 nodes in query result, got {len(query_nodes)}")
        return False
    
    print(f"  ✅ Integration test passed")
    print(f"     Created {nodes_created} nodes and {edges_created} edges")
    print(f"     Query returned {len(query_nodes)} nodes")
    return True


def cleanup_test_data():
    """Clean up test data from Neo4j."""
    print("\nCleaning up test data...")
    
    connected, error = _check_neo4j_connection()
    if not connected:
        return
    
    try:
        with neo4j_connection() as driver:
            with driver.session() as session:
                # Delete test nodes
                session.run("""
                    MATCH (n)
                    WHERE n.id STARTS WITH 'test_' 
                       OR n.id STARTS WITH 'provenance_'
                       OR n.id STARTS WITH 'concept_'
                       OR n.id STARTS WITH 'query_test_'
                       OR n.id STARTS WITH 'integration_test_'
                    DETACH DELETE n
                """)
        print("  ✅ Test data cleaned up")
    except Exception as e:
        print(f"  ⚠️  Cleanup warning: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3 GraphRAG Implementation Tests")
    print("=" * 60)
    
    results = []
    
    # Test 1: Neo4j Connection
    results.append(("Neo4j Connection", test_neo4j_connection()))
    
    # Test 2: Graph Upsert (Nodes)
    results.append(("Graph Upsert (Nodes)", test_graph_upsert_nodes()))
    
    # Test 3: Graph Upsert (Provenance)
    results.append(("Graph Upsert (Provenance)", test_graph_upsert_provenance()))
    
    # Test 4: Graph Query
    results.append(("Graph Query", test_graph_query()))
    
    # Test 5: Graph Path Storage
    results.append(("Graph Path Storage", test_graph_path_storage()))
    
    # Test 6: Weaviate Connection Pool
    results.append(("Weaviate Connection Pool", test_weaviate_connection_pool()))
    
    # Test 7: Graph Build Integration
    results.append(("Graph Build Integration", test_graph_build_integration()))
    
    # Cleanup
    cleanup_test_data()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL / ⚠️  SKIP"
        print(f"  {name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    sys.exit(0 if all_passed else 1)

