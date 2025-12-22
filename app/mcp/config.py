# app/mcp/config.py
"""
Centralized configuration module for MCP API keys and settings.

This module provides:
- Automatic .env file loading
- Centralized access to all API keys
- Validation and error handling
- Support for both .env files and direct environment variables
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False
    load_dotenv = None  # type: ignore[assignment]


class MCPConfig:
    """Centralized configuration for MCP API keys and settings."""
    
    def __init__(self):
        """Initialize configuration by loading .env file and setting up API keys."""
        self._load_dotenv()
        self._validate_required_keys()
    
    def _load_dotenv(self) -> None:
        """Load .env file if python-dotenv is available."""
        if _DOTENV_AVAILABLE and load_dotenv:
            # Try to load .env from current working directory
            env_path = Path.cwd() / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=str(env_path))
                print(f"Loaded .env file from: {env_path}")
            else:
                # Try to load .env from project root (where this config is located)
                project_root = Path(__file__).parent.parent.parent
                env_path = project_root / ".env"
                if env_path.exists():
                    load_dotenv(dotenv_path=str(env_path))
                    print(f"Loaded .env file from: {env_path}")
                else:
                    print("No .env file found, using environment variables only")
        elif not _DOTENV_AVAILABLE:
            print("python-dotenv not available, using environment variables only")
    
    def _validate_required_keys(self) -> None:
        """Validate that required API keys are available."""
        required_keys = [
            "OPENAI_API_KEY",
            # "WEAVIATE_HOST", 
            # "WEAVIATE_PORT",
            # "WEAVIATE_SCHEME"
        ]
        
        missing_keys = []
        for key in required_keys:
            if not self.get(key):
                missing_keys.append(key)
        
        if missing_keys:
            print(f"Warning: Missing required API keys: {', '.join(missing_keys)}")
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get an environment variable value.
        
        Args:
            key: Environment variable name
            default: Default value if key is not found
            
        Returns:
            Value of the environment variable or default
        """
        return os.getenv(key, default)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """
        Get an environment variable as integer.
        
        Args:
            key: Environment variable name
            default: Default value if key is not found or invalid
            
        Returns:
            Integer value of the environment variable
        """
        try:
            value = self.get(key)
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """
        Get an environment variable as boolean.
        
        Args:
            key: Environment variable name
            default: Default value if key is not found or invalid
            
        Returns:
            Boolean value of the environment variable
        """
        try:
            value = self.get(key)
            if value is None:
                return default
            return value.lower() in ('true', '1', 'yes', 'on')
        except (AttributeError, TypeError):
            return default
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """OpenAI API key for embeddings."""
        return self.get("OPENAI_API_KEY")
    
    @property
    def deepseek_api_key(self) -> Optional[str]:
        """DeepSeek API key."""
        return self.get("DEEPSEEK_API_KEY")
    
    @property
    def google_api_key(self) -> Optional[str]:
        """Google API key."""
        return self.get("GOOGLE_API_KEY")
    
    @property
    def groq_api_key(self) -> Optional[str]:
        """Groq API key."""
        return self.get("GROQ_API_KEY")
    
    @property
    def anthropic_api_key(self) -> Optional[str]:
        """Anthropic API key."""
        return self.get("ANTHROPIC_API_KEY")
    
    @property
    def tavily_api_key(self) -> Optional[str]:
        """Tavily API key."""
        return self.get("TAVILY_API_KEY")
    
    @property
    def searxng_secret(self) -> Optional[str]:
        """SearXNG secret."""
        return self.get("SEARXNG_SECRET")
    
    @property
    def weaviate_host(self) -> str:
        """Weaviate host."""
        return self.get("WEAVIATE_HOST", "localhost")
    
    @property
    def weaviate_port(self) -> int:
        """Weaviate port."""
        return self.get_int("WEAVIATE_PORT", 22006)
    
    @property
    def weaviate_scheme(self) -> str:
        """Weaviate scheme (http/https)."""
        return self.get("WEAVIATE_SCHEME", "http")
    
    @property
    def weaviate_url(self) -> str:
        """Weaviate URL."""
        return f"{self.weaviate_scheme}://{self.weaviate_host}:{self.weaviate_port}"
    
    @property
    def test_pdf_path(self) -> Optional[str]:
        """Test PDF path."""
        return self.get("TEST_PDF_PATH")
    
    @property
    def weaviate_data_path(self) -> Optional[str]:
        """Weaviate data path."""
        return self.get("WEAVIATE_DATA_PATH")
    
    def get_all_api_keys(self) -> Dict[str, Optional[str]]:
        """Get all API keys as a dictionary."""
        return {
            "OPENAI_API_KEY": self.openai_api_key,
            "DEEPSEEK_API_KEY": self.deepseek_api_key,
            "GOOGLE_API_KEY": self.google_api_key,
            "GROQ_API_KEY": self.groq_api_key,
            "ANTHROPIC_API_KEY": self.anthropic_api_key,
            "TAVILY_API_KEY": self.tavily_api_key,
            "SEARXNG_SECRET": self.searxng_secret,
        }
    
    def get_weaviate_config(self) -> Dict[str, Any]:
        """Get Weaviate configuration as a dictionary."""
        return {
            "host": self.weaviate_host,
            "port": self.weaviate_port,
            "scheme": self.weaviate_scheme,
            "url": self.weaviate_url,
        }


# Global configuration instance
_config = MCPConfig()


def get_config() -> MCPConfig:
    """Get the global MCP configuration instance."""
    return _config


def get_api_key(key_name: str) -> Optional[str]:
    """
    Get an API key by name.
    
    Args:
        key_name: Name of the API key (e.g., "OPENAI_API_KEY")
        
    Returns:
        API key value or None if not found
    """
    return _config.get(key_name)


def get_weaviate_client_config() -> Dict[str, Any]:
    """Get Weaviate client configuration."""
    return _config.get_weaviate_config()


# Convenience functions for direct access
def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key."""
    return _config.openai_api_key


def get_deepseek_api_key() -> Optional[str]:
    """Get DeepSeek API key."""
    return _config.deepseek_api_key


def get_google_api_key() -> Optional[str]:
    """Get Google API key."""
    return _config.google_api_key


def get_groq_api_key() -> Optional[str]:
    """Get Groq API key."""
    return _config.groq_api_key


def get_anthropic_api_key() -> Optional[str]:
    """Get Anthropic API key."""
    return _config.anthropic_api_key


def get_tavily_api_key() -> Optional[str]:
    """Get Tavily API key."""
    return _config.tavily_api_key


def get_searxng_secret() -> Optional[str]:
    """Get SearXNG secret."""
    return _config.searxng_secret
