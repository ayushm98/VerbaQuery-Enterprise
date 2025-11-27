import os
import tempfile
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Try to import streamlit secrets for cloud deployment
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def get_api_key() -> str:
    """Get OpenAI API key from Streamlit secrets or environment."""
    # First try Streamlit secrets (for cloud deployment)
    if HAS_STREAMLIT:
        try:
            return st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass
    # Fall back to environment variable
    return os.getenv("OPENAI_API_KEY", "")


def is_cloud_environment() -> bool:
    """Detect if running in Streamlit Cloud or similar cloud environment."""
    # Check for common cloud environment indicators
    return (
        os.getenv("STREAMLIT_SHARING_MODE") is not None or
        os.getenv("STREAMLIT_CLOUD") is not None or
        HAS_STREAMLIT and hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets
    )


def get_writable_dir() -> Path:
    """Get a writable directory for cloud environments."""
    if is_cloud_environment():
        # Use /tmp for cloud deployments (writable directory)
        return Path(tempfile.gettempdir())
    else:
        # Use local ./data for local development
        return Path("./data")


class Settings(BaseSettings):
    """
    Centralized configuration management using Pydantic.
    Loads from environment variables, .env file, or Streamlit secrets.
    """

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4-turbo-preview"

    # Chunking Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval Configuration
    initial_retrieval_count: int = 10
    final_retrieval_count: int = 5
    ensemble_weight_vector: float = 0.5
    ensemble_weight_keyword: float = 0.5

    # Database Configuration
    chroma_persist_directory: Path = Path("./data/indexes/chroma")
    bm25_index_path: Path = Path("./data/indexes/bm25_index.pkl")

    # Upload Configuration
    max_upload_size_mb: int = 50

    # Application Configuration
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Use writable directory in cloud environments
        base_dir = get_writable_dir()

        # Update paths to use writable directory
        self.chroma_persist_directory = base_dir / "indexes" / "chroma"
        self.bm25_index_path = base_dir / "indexes" / "bm25_index.pkl"

        # Ensure directory paths exist
        self.chroma_persist_directory.parent.mkdir(parents=True, exist_ok=True)
        self.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """
    Singleton pattern for settings.
    Cached to avoid repeated environment variable parsing.
    Supports both .env files and Streamlit secrets.
    """
    # Get API key from Streamlit secrets or environment
    api_key = get_api_key()
    if api_key:
        return Settings(openai_api_key=api_key)
    return Settings()
