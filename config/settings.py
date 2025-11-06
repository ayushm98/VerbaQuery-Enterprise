from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized configuration management using Pydantic.
    Loads from environment variables and .env file.
    """

    # OpenAI Configuration
    openai_api_key: str
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

    # Application Configuration
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directory paths exist
        self.chroma_persist_directory.parent.mkdir(parents=True, exist_ok=True)
        self.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """
    Singleton pattern for settings.
    Cached to avoid repeated environment variable parsing.
    """
    return Settings()
