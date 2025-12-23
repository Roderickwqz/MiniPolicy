# app/mcp/tools/neo4j_client.py
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import neo4j
from neo4j import GraphDatabase

from app.mcp.config import get_neo4j_config


# Thread-local storage for connection reuse
import threading
_thread_local = threading.local()


def get_neo4j_driver() -> Optional[neo4j.Driver]:
    """
    Get a Neo4j driver instance.
    Returns None if configuration is missing.
    """
    config = get_neo4j_config()
    uri = config.get("uri")
    username = config.get("username")
    password = config.get("password")
    
    if not uri or not username or not password:
        return None
    
    return GraphDatabase.driver(uri, auth=(username, password))


@contextmanager
def neo4j_connection():
    """
    Context manager for Neo4j connection pooling.
    Reuses driver within the same thread context.
    
    Usage:
        with neo4j_connection() as driver:
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN n LIMIT 1")
                # Driver is automatically closed when exiting context
    """
    # Check if we already have a driver in this thread
    if hasattr(_thread_local, 'neo4j_driver'):
        driver = _thread_local.neo4j_driver
        try:
            # Verify driver is still alive
            driver.verify_connectivity()
            yield driver
            return
        except Exception:
            # Driver is dead, create a new one
            try:
                driver.close()
            except Exception:
                pass
            delattr(_thread_local, 'neo4j_driver')
    
    # Create new driver
    driver = get_neo4j_driver()
    if driver is None:
        raise ValueError(
            "Neo4j configuration missing. Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables."
        )
    
    # Store in thread-local for potential reuse
    _thread_local.neo4j_driver = driver
    
    try:
        yield driver
    finally:
        # Clean up: close driver and remove from thread-local
        try:
            driver.close()
        except Exception:
            pass
        if hasattr(_thread_local, 'neo4j_driver'):
            delattr(_thread_local, 'neo4j_driver')

