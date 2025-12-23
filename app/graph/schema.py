# app/graph/schema.py
"""
Graph schema definitions for Neo4j GraphRAG.

Node types and relationship types used in the knowledge graph.
"""

from typing import Any, Dict, List, Literal, Optional

# Node types
NodeType = Literal["Doc", "Chunk", "Concept", "Obligation", "Risk", "Penalty"]

# Relationship types
RelationshipType = Literal[
    "HAS_CHUNK",      # Doc -> Chunk
    "DERIVED_FROM",   # Derived nodes (Concept, Obligation, Risk, Penalty) -> Chunk (provenance)
    "MENTIONS",       # Node mentions another node
    "REQUIRES",       # Node requires another node
    "TRIGGERS",       # Node triggers another node
    "SUBJECT_TO",     # Node is subject to another node
    "LEADS_TO",       # Node leads to another node
    "MITIGATES",      # Node mitigates another node
    "SUPPORTS",       # Node supports another node
]

# Node property schemas
NODE_PROPERTIES: Dict[NodeType, List[str]] = {
    "Doc": [
        "doc_id",      # Unique document identifier
        "doc_hash",    # Document content hash
        "source_path", # Source file path
        "page_count",  # Number of pages
    ],
    "Chunk": [
        "chunk_id",    # Unique chunk identifier
        "content_id", # Content hash (for deduplication)
        "text",        # Chunk text content
        "doc_id",      # Parent document ID
        "page_num",    # Page number
        "page_index",  # Page index (0-based)
        "chunk_index_global",  # Global chunk index
        "chunk_index_in_page", # Chunk index within page
        "chunk_method", # Chunking method used
        "approx_tokens", # Approximate token count
    ],
    "Concept": [
        "concept_id",  # Unique concept identifier
        "name",        # Concept name
        "description", # Concept description
        "confidence",  # Confidence score
    ],
    "Obligation": [
        "obligation_id", # Unique obligation identifier
        "title",         # Obligation title
        "description",   # Obligation description
        "confidence",    # Confidence score
    ],
    "Risk": [
        "risk_id",      # Unique risk identifier
        "title",        # Risk title
        "description",  # Risk description
        "severity",     # Risk severity level
        "confidence",   # Confidence score
    ],
    "Penalty": [
        "penalty_id",   # Unique penalty identifier
        "title",        # Penalty title
        "description",  # Penalty description
        "amount",       # Penalty amount (if applicable)
        "confidence",   # Confidence score
    ],
}

# Relationship property schemas
RELATIONSHIP_PROPERTIES: Dict[RelationshipType, List[str]] = {
    "HAS_CHUNK": [
        "order",       # Order of chunk in document
    ],
    "DERIVED_FROM": [
        "confidence",  # Confidence in derivation
        "method",      # Method used for derivation
    ],
    "MENTIONS": [
        "context",     # Context of mention
    ],
    "REQUIRES": [
        "condition",   # Condition for requirement
    ],
    "TRIGGERS": [
        "condition",  # Trigger condition
    ],
    "SUBJECT_TO": [
        "scope",      # Scope of subjection
    ],
    "LEADS_TO": [
        "probability", # Probability of leading to
    ],
    "MITIGATES": [
        "effectiveness", # Mitigation effectiveness
    ],
    "SUPPORTS": [
        "strength",   # Support strength
    ],
}


def get_node_label(node_type: NodeType) -> str:
    """Get Neo4j label for node type."""
    return node_type


def get_relationship_type(rel_type: RelationshipType) -> str:
    """Get Neo4j relationship type."""
    return rel_type


def validate_node_properties(node_type: NodeType, properties: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize node properties.
    Returns cleaned properties dict.
    """
    allowed = NODE_PROPERTIES.get(node_type, [])
    cleaned = {}
    for key, value in properties.items():
        if key in allowed:
            cleaned[key] = value
    return cleaned


def validate_relationship_properties(
    rel_type: RelationshipType, properties: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate and normalize relationship properties.
    Returns cleaned properties dict.
    """
    allowed = RELATIONSHIP_PROPERTIES.get(rel_type, [])
    cleaned = {}
    for key, value in properties.items():
        if key in allowed:
            cleaned[key] = value
    return cleaned

