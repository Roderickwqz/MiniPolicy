#!/usr/bin/env python3
"""
Test script to verify that dotenv loading is working correctly.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_config():
    """Test the centralized configuration module."""
    print("Testing MCP Configuration...")
    
    try:
        from app.mcp.config import get_config, get_openai_api_key, get_weaviate_client_config
        
        # Get the global config instance
        config = get_config()
        
        print("\n=== Configuration Test Results ===")
        
        # Test API keys
        print(f"OpenAI API Key: {'✓ Found' if config.openai_api_key else '✗ Not found'}")
        print(f"DeepSeek API Key: {'✓ Found' if config.deepseek_api_key else '✗ Not found'}")
        print(f"Google API Key: {'✓ Found' if config.google_api_key else '✗ Not found'}")
        print(f"Groq API Key: {'✓ Found' if config.groq_api_key else '✗ Not found'}")
        print(f"Anthropic API Key: {'✓ Found' if config.anthropic_api_key else '✗ Not found'}")
        print(f"Tavily API Key: {'✓ Found' if config.tavily_api_key else '✗ Not found'}")
        print(f"SearXNG Secret: {'✓ Found' if config.searxng_secret else '✗ Not found'}")
        
        # Test Weaviate configuration
        weaviate_config = config.get_weaviate_config()
        print(f"\nWeaviate Configuration:")
        print(f"  Host: {weaviate_config['host']}")
        print(f"  Port: {weaviate_config['port']}")
        print(f"  Scheme: {weaviate_config['scheme']}")
        print(f"  URL: {weaviate_config['url']}")
        
        # Test convenience functions
        print(f"\nConvenience Functions:")
        print(f"OpenAI API Key (direct): {'✓ Found' if get_openai_api_key() else '✗ Not found'}")
        
        # Test all API keys
        all_keys = config.get_all_api_keys()
        print(f"\nAll API Keys Summary:")
        for key, value in all_keys.items():
            print(f"  {key}: {'✓ Found' if value else '✗ Not found'}")
        
        print("\n=== Test Completed Successfully ===")
        return True
        
    except ImportError as e:
        print(f"Import Error: {e}")
        return False
    except Exception as e:
        print(f"Test Error: {e}")
        return False

if __name__ == "__main__":
    success = test_config()
    sys.exit(0 if success else 1)
