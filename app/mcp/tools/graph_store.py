# app/mcp/tools/graph_store.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import hashlib
import uuid

from app.mcp.contracts import ToolError, ToolResult
from app.mcp.tools.neo4j_client import neo4j_connection
from app.graph.schema import (
    NodeType,
    RelationshipType,
    get_node_label,
    get_relationship_type,
    validate_node_properties,
    validate_relationship_properties,
)


def _generate_node_id(node_type: NodeType, identifier: str) -> str:
    """Generate a stable node ID from type and identifier."""
    return f"{node_type}:{identifier}"


def _generate_graph_path_id(query: str, result_hash: Optional[str] = None) -> str:
    """Generate a unique graph_path_id for query results."""
    if result_hash:
        return f"graph_path::{result_hash}"
    # Use query hash as fallback
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return f"graph_path::{query_hash}"


def graph_upsert_tool(args: Dict[str, Any]) -> ToolResult:
    """
    Upsert nodes and edges into Neo4j graph.
    
    args:
      - nodes: list of node dicts with:
          - type: NodeType (Doc, Chunk, Concept, Obligation, Risk, Penalty)
          - id: unique identifier (e.g., doc_id, chunk_id)
          - properties: dict of node properties
      - edges: list of edge dicts with:
          - from_type: NodeType
          - from_id: source node identifier
          - to_type: NodeType
          - to_id: target node identifier
          - relationship: RelationshipType
          - properties: dict of relationship properties (optional)
    
    returns:
      - nodes_created: number of nodes created
      - nodes_updated: number of nodes updated
      - edges_created: number of edges created
      - edges_updated: number of edges updated
    """
    tool_name = "graph_upsert_tool"
    
    nodes = args.get("nodes", [])
    edges = args.get("edges", [])
    
    if not isinstance(nodes, list):
        return ToolResult(
            ok=False,
            tool_name=tool_name,
            args=args,
            error=ToolError(code="VALIDATION_ERROR", message="nodes must be a list"),
        )
    
    if not isinstance(edges, list):
        return ToolResult(
            ok=False,
            tool_name=tool_name,
            args=args,
            error=ToolError(code="VALIDATION_ERROR", message="edges must be a list"),
        )
    
    try:
        with neo4j_connection() as driver:
            with driver.session() as session:
                nodes_created = 0
                nodes_updated = 0
                edges_created = 0
                edges_updated = 0
                
                # Upsert nodes
                for node_data in nodes:
                    node_type = node_data.get("type")
                    node_id = node_data.get("id")
                    properties = node_data.get("properties", {})
                    
                    if not node_type or not node_id:
                        continue
                    
                    # Validate node type
                    if node_type not in ["Doc", "Chunk", "Concept", "Obligation", "Risk", "Penalty"]:
                        continue
                    
                    # Validate and clean properties
                    cleaned_props = validate_node_properties(node_type, properties)
                    cleaned_props["id"] = node_id  # Ensure id is in properties
                    
                    # Use MERGE to create or update node
                    label = get_node_label(node_type)
                    cypher = f"""
                    MERGE (n:{label} {{id: $id}})
                    SET n += $properties
                    RETURN n
                    """
                    
                    result = session.run(cypher, id=node_id, properties=cleaned_props)
                    record = result.single()
                    if record:
                        # Check if node was created or updated
                        # MERGE always returns a node, so we check if it had existing properties
                        # For simplicity, we count as created if it's a new node
                        # In practice, you might want to track this more carefully
                        nodes_created += 1
                
                # Upsert edges
                for edge_data in edges:
                    from_type = edge_data.get("from_type")
                    from_id = edge_data.get("from_id")
                    to_type = edge_data.get("to_type")
                    to_id = edge_data.get("to_id")
                    relationship = edge_data.get("relationship")
                    rel_properties = edge_data.get("properties", {})
                    
                    if not all([from_type, from_id, to_type, to_id, relationship]):
                        continue
                    
                    # Validate relationship type
                    valid_rels = [
                        "HAS_CHUNK", "DERIVED_FROM", "MENTIONS", "REQUIRES",
                        "TRIGGERS", "SUBJECT_TO", "LEADS_TO", "MITIGATES", "SUPPORTS"
                    ]
                    if relationship not in valid_rels:
                        continue
                    
                    # Validate and clean relationship properties
                    cleaned_rel_props = validate_relationship_properties(relationship, rel_properties)
                    
                    # Use MERGE to create or update relationship
                    from_label = get_node_label(from_type)
                    to_label = get_node_label(to_type)
                    rel_type = get_relationship_type(relationship)
                    
                    cypher = f"""
                    MATCH (from:{from_label} {{id: $from_id}}), (to:{to_label} {{id: $to_id}})
                    MERGE (from)-[r:{rel_type}]->(to)
                    SET r += $properties
                    RETURN r
                    """
                    
                    result = session.run(
                        cypher,
                        from_id=from_id,
                        to_id=to_id,
                        properties=cleaned_rel_props if cleaned_rel_props else {},
                    )
                    record = result.single()
                    if record:
                        edges_created += 1
                
                return ToolResult(
                    ok=True,
                    tool_name=tool_name,
                    args=args,
                    data={
                        "nodes_created": nodes_created,
                        "nodes_updated": nodes_updated,
                        "edges_created": edges_created,
                        "edges_updated": edges_updated,
                    },
                )
    
    except ValueError as e:
        # Configuration error
        return ToolResult(
            ok=False,
            tool_name=tool_name,
            args=args,
            error=ToolError(code="CONFIGURATION_ERROR", message=str(e)),
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            tool_name=tool_name,
            args=args,
            error=ToolError(code="TOOL_RUNTIME_ERROR", message=str(e)),
        )


def graph_query_tool(args: Dict[str, Any]) -> ToolResult:
    """
    Execute a Cypher query on Neo4j and return graph_path_id with path snapshot.
    
    args:
      - query: str (Cypher query, required)
      - run_dir: str (optional, for storing path snapshot in artifacts.json)
      - run_id: str (optional, used to construct run_dir if run_dir not provided)
    
    returns:
      - graph_path_id: str (unique identifier for this query result)
      - nodes: list of matched nodes
      - edges: list of matched edges/relationships
      - query: the original query
    """
    tool_name = "graph_query_tool"
    
    query = args.get("query", "").strip()
    if not query:
        return ToolResult(
            ok=False,
            tool_name=tool_name,
            args=args,
            error=ToolError(code="VALIDATION_ERROR", message="query is required"),
        )
    
    # Determine run_dir from args
    run_dir = args.get("run_dir")
    if not run_dir:
        run_id = args.get("run_id")
        if run_id:
            run_dir = f"app/artifacts/{run_id}"
    
    try:
        with neo4j_connection() as driver:
            with driver.session() as session:
                # Execute query
                result = session.run(query)
                
                # Collect nodes and edges from result
                nodes: List[Dict[str, Any]] = []
                edges: List[Dict[str, Any]] = []
                node_ids_seen: set = set()
                edge_ids_seen: set = set()
                
                for record in result:
                    # Extract nodes and edges from record
                    for key, value in record.items():
                        # Handle Path objects (contains nodes and relationships)
                        if hasattr(value, 'nodes') and hasattr(value, 'relationships'):
                            # Path object
                            for node in value.nodes:
                                node_id = node.get("id") if hasattr(node, 'get') else (str(node.id) if hasattr(node, 'id') else None)
                                if not node_id:
                                    continue
                                if node_id not in node_ids_seen:
                                    node_ids_seen.add(node_id)
                                    nodes.append({
                                        "id": node_id,
                                        "labels": list(node.labels) if hasattr(node, 'labels') else [],
                                        "properties": dict(node) if hasattr(node, '__iter__') else {},
                                    })
                            # Extract relationships from path
                            for rel in value.relationships:
                                rel_id = str(rel.id) if hasattr(rel, 'id') else None
                                if rel_id and rel_id not in edge_ids_seen:
                                    edge_ids_seen.add(rel_id)
                                    start_id = rel.start_node.get("id") if hasattr(rel.start_node, 'get') else (str(rel.start_node.id) if hasattr(rel.start_node, 'id') else None)
                                    end_id = rel.end_node.get("id") if hasattr(rel.end_node, 'get') else (str(rel.end_node.id) if hasattr(rel.end_node, 'id') else None)
                                    if start_id and end_id:
                                        edges.append({
                                            "id": rel_id,
                                            "type": rel.type if hasattr(rel, 'type') else "",
                                            "start_node": start_id,
                                            "end_node": end_id,
                                            "properties": dict(rel) if hasattr(rel, '__iter__') else {},
                                        })
                        # Handle Node objects
                        elif hasattr(value, 'labels') and hasattr(value, 'id'):
                            node_id = value.get("id") if hasattr(value, 'get') else (str(value.id) if hasattr(value, 'id') else None)
                            if node_id and node_id not in node_ids_seen:
                                node_ids_seen.add(node_id)
                                nodes.append({
                                    "id": node_id,
                                    "labels": list(value.labels) if hasattr(value, 'labels') else [],
                                    "properties": dict(value) if hasattr(value, '__iter__') else {},
                                })
                        # Handle Relationship objects
                        elif hasattr(value, 'type') and hasattr(value, 'start_node'):
                            rel_id = str(value.id) if hasattr(value, 'id') else None
                            if rel_id and rel_id not in edge_ids_seen:
                                edge_ids_seen.add(rel_id)
                                start_id = value.start_node.get("id") if hasattr(value.start_node, 'get') else (str(value.start_node.id) if hasattr(value.start_node, 'id') else None)
                                end_id = value.end_node.get("id") if hasattr(value.end_node, 'get') else (str(value.end_node.id) if hasattr(value.end_node, 'id') else None)
                                if start_id and end_id:
                                    edges.append({
                                        "id": rel_id,
                                        "type": value.type if hasattr(value, 'type') else "",
                                        "start_node": start_id,
                                        "end_node": end_id,
                                        "properties": dict(value) if hasattr(value, '__iter__') else {},
                                    })
                
                # Generate graph_path_id from query and result hash
                result_hash = hashlib.sha256(
                    (query + str(len(nodes)) + str(len(edges))).encode("utf-8")
                ).hexdigest()[:16]
                graph_path_id = _generate_graph_path_id(query, result_hash)
                
                # Store path snapshot in artifacts.json if run_dir provided
                if run_dir:
                    from app.mcp.artifacts import append_graph_path
                    append_graph_path(
                        run_dir=run_dir,
                        graph_path_id=graph_path_id,
                        query=query,
                        nodes=nodes,
                        edges=edges,
                    )
                
                return ToolResult(
                    ok=True,
                    tool_name=tool_name,
                    args=args,
                    data={
                        "graph_path_id": graph_path_id,
                        "nodes": nodes,
                        "edges": edges,
                        "query": query,
                        "node_count": len(nodes),
                        "edge_count": len(edges),
                    },
                )
    
    except ValueError as e:
        # Configuration error
        return ToolResult(
            ok=False,
            tool_name=tool_name,
            args=args,
            error=ToolError(code="CONFIGURATION_ERROR", message=str(e)),
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            tool_name=tool_name,
            args=args,
            error=ToolError(code="TOOL_RUNTIME_ERROR", message=str(e)),
        )
