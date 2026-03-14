import os

import torch
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from functools import lru_cache

load_dotenv(override=True)
class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    HF_AUTH_KEY: str = os.getenv("HF_AUTH_KEY")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL")

    EMBEDDING_MODEL_PATH: str = f"./embedding_models/{EMBEDDING_MODEL}"

    MAX_TOKEN: int = int(os.getenv("MAX_TOKEN"))
    # Qdrant
    QDRANT_HOST: str = os.getenv("QDRANT_HOST")
    QDRANT_PORT: int = int(str(os.getenv("QDRANT_PORT")))
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME")
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE"))
    # RAG Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD"))

    # Agent Configuration
    MAX_ITERATIONS: int = os.getenv("MAX_ITERATIONS")
    ENABLE_SELF_REFLECTION: bool = os.getenv("ENABLE_SELF_REFLECTION")
    BATCH_SIZE: int = os.getenv("BATCH_SIZE")
    # API Configuration
    API_HOST: str = os.getenv("API_HOST")
    API_PORT: int = os.getenv("API_PORT")

    DEVICE: str = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"

    class Config:
        env_file = ".env"
        case_sensitive = True

app_settings = Settings()