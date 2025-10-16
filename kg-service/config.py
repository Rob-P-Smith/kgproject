"""
Configuration management for KG Service
Handles environment variables and service settings
"""

import os
import logging.config
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Service Configuration
    SERVICE_NAME: str = "kg-service"
    SERVICE_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8088, env="API_PORT")
    API_PREFIX: str = "/api/v1"

    # Neo4j Configuration
    NEO4J_URI: str = Field(default="bolt://neo4j-kg:7687", env="NEO4J_URI")
    NEO4J_USER: str = Field(default="neo4j", env="NEO4J_USER")
    NEO4J_PASSWORD: str = Field(default="knowledge_graph_2024", env="NEO4J_PASSWORD")
    NEO4J_DATABASE: str = Field(default="neo4j", env="NEO4J_DATABASE")
    NEO4J_MAX_CONNECTION_LIFETIME: int = 3600
    NEO4J_MAX_CONNECTION_POOL_SIZE: int = 50
    NEO4J_CONNECTION_TIMEOUT: int = 30

    # vLLM Configuration
    VLLM_BASE_URL: str = Field(default="http://localhost:8078", env="VLLM_BASE_URL")
    VLLM_MODEL_NAME: Optional[str] = None  # Discovered at runtime
    VLLM_TIMEOUT: int = Field(
        default=600,
        env="VLLM_TIMEOUT"
    )  # seconds (relationship extraction needs more time)
    VLLM_MAX_TOKENS: int = 4096
    VLLM_TEMPERATURE: float = 0.1  # Low temperature for consistent extraction
    VLLM_RETRY_INTERVAL: int = 30  # seconds between retries
    VLLM_MAX_RETRIES: int = 3

    # GLiNER Configuration
    GLINER_MODEL: str = Field(
        default="urchade/gliner_large-v2.1",
        env="GLINER_MODEL"
    )
    GLINER_THRESHOLD: float = Field(
        default=0.4,
        env="GLINER_THRESHOLD"
    )  # Confidence threshold
    GLINER_BATCH_SIZE: int = 8
    GLINER_MAX_LENGTH: int = 384  # tokens

    # Entity Extraction Settings
    ENTITY_TAXONOMY_PATH: str = "taxonomy/entities.yaml"
    ENTITY_MIN_CONFIDENCE: float = Field(
        default=0.4,
        env="GLINER_THRESHOLD"
    )  # Use same threshold as GLiNER
    ENTITY_DEDUPLICATION: bool = True

    # Relationship Extraction Settings
    RELATION_MIN_CONFIDENCE: float = Field(
        default=0.45,
        env="RELATION_MIN_CONFIDENCE"
    )  # Relationship confidence threshold
    RELATION_MAX_DISTANCE: int = 3  # Max sentence distance between entities
    RELATION_CONTEXT_WINDOW: int = 200  # characters

    # Co-occurrence Settings
    COOCCURRENCE_WINDOW: int = 100  # characters
    COOCCURRENCE_MIN_COUNT: int = 2

    # Processing Settings
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 300  # seconds
    ENABLE_ASYNC_PROCESSING: bool = True

    # Document Processing
    MAX_DOCUMENT_LENGTH: int = 100000  # characters
    CHUNK_SIZE: int = 2000  # characters for processing large documents
    CHUNK_OVERLAP: int = 200  # characters

    # Cache Settings
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 3600  # seconds
    CACHE_MAX_SIZE: int = 1000  # items

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60

    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings singleton"""
    return settings


def validate_settings() -> bool:
    """Validate critical settings on startup"""
    errors = []

    # Check Neo4j configuration
    if not settings.NEO4J_URI:
        errors.append("NEO4J_URI is not set")
    if not settings.NEO4J_PASSWORD:
        errors.append("NEO4J_PASSWORD is not set")

    # Check vLLM configuration
    if not settings.VLLM_BASE_URL:
        errors.append("VLLM_BASE_URL is not set")

    # Check file paths
    taxonomy_path = os.path.join(
        os.path.dirname(__file__),
        settings.ENTITY_TAXONOMY_PATH
    )
    if not os.path.exists(taxonomy_path):
        errors.append(f"Entity taxonomy file not found at {taxonomy_path}")

    if errors:
        for error in errors:
            print(f"Configuration Error: {error}")
        return False

    return True


# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": settings.LOG_LEVEL,
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": settings.LOG_LEVEL,
            "formatter": "detailed",
            "filename": "kg-service.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "": {
            "level": settings.LOG_LEVEL,
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
        "fastapi": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
    },
}


if __name__ == "__main__":
    # Test configuration
    print("Configuration Settings:")
    print(f"Service: {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")
    print(f"API: {settings.API_HOST}:{settings.API_PORT}")
    print(f"Neo4j: {settings.NEO4J_URI}")
    print(f"vLLM: {settings.VLLM_BASE_URL}")
    print(f"GLiNER Model: {settings.GLINER_MODEL}")
    print(f"\nValidation: {'✓ PASS' if validate_settings() else '✗ FAIL'}")
