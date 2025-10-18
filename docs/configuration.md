---
layout: default
title: Configuration & Clients
---

# Configuration & Clients Documentation

## Module Overview

The configuration module centralizes all service settings using Pydantic for type-safe configuration management with environment variable support. The clients module provides interfaces to external services (vLLM inference server).

**Files**:
- `config.py`: Service configuration and logging setup
- `clients/vllm_client.py`: vLLM HTTP client with auto-discovery and retry logic
- `clients/__init__.py`: Module exports

## Configuration

### Purpose

Centralize all service configuration parameters with environment variable support, validation, and sensible defaults. Uses Pydantic Settings for type safety and automatic environment loading.

### Architecture

**Class**: `Settings (extends BaseSettings)`

**Configuration Sources** (priority order):
1. Environment variables (.env file or system environment)
2. Default values in Settings class

**Global Instance**: `settings = Settings()` (loaded on module import)

### Configuration Categories

#### Service Configuration

```python
SERVICE_NAME: str = "kg-service"
SERVICE_VERSION: str = "1.0.0"
DEBUG: bool = Field(default=False, env="DEBUG")
LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
```

**Purpose**: Basic service identification and runtime mode

**Environment Variables**:
- `DEBUG`: Enable debug mode (default: False)
- `LOG_LEVEL`: Logging level (default: INFO, options: DEBUG, INFO, WARNING, ERROR)

**Usage**:
- `SERVICE_NAME`, `SERVICE_VERSION`: Returned in health/stats endpoints
- `DEBUG`: Enables auto-reload in uvicorn, verbose logging
- `LOG_LEVEL`: Controls logging verbosity across all modules

#### API Configuration

```python
API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
API_PORT: int = Field(default=8088, env="API_PORT")
API_PREFIX: str = "/api/v1"
```

**Purpose**: HTTP server binding configuration

**Environment Variables**:
- `API_HOST`: Bind address (default: 0.0.0.0, all interfaces)
- `API_PORT`: Port number (default: 8088)

**API_PREFIX**: Base path for versioned endpoints (not configurable via env)

**Usage**:
- Uvicorn server binds to `API_HOST:API_PORT`
- All API endpoints prefixed with `/api/v1`

#### Neo4j Configuration

```python
NEO4J_URI: str = Field(default="bolt://neo4j-kg:7687", env="NEO4J_URI")
NEO4J_USER: str = Field(default="neo4j", env="NEO4J_USER")
NEO4J_PASSWORD: str = Field(default="knowledge_graph_2024", env="NEO4J_PASSWORD")
NEO4J_DATABASE: str = Field(default="neo4j", env="NEO4J_DATABASE")
NEO4J_MAX_CONNECTION_LIFETIME: int = 3600
NEO4J_MAX_CONNECTION_POOL_SIZE: int = 50
NEO4J_CONNECTION_TIMEOUT: int = 30
```

**Purpose**: Neo4j database connection parameters

**Environment Variables**:
- `NEO4J_URI`: Bolt protocol URI (default: bolt://neo4j-kg:7687)
- `NEO4J_USER`: Database username (default: neo4j)
- `NEO4J_PASSWORD`: Database password (default: knowledge_graph_2024)
- `NEO4J_DATABASE`: Database name (default: neo4j)

**Connection Pool Settings** (not configurable via env):
- `MAX_CONNECTION_LIFETIME`: 3600 seconds (1 hour)
- `MAX_CONNECTION_POOL_SIZE`: 50 connections
- `CONNECTION_TIMEOUT`: 30 seconds

**Security Note**: Always override `NEO4J_PASSWORD` in production via environment variable

#### vLLM Configuration

```python
VLLM_BASE_URL: str = Field(default="http://localhost:8078", env="VLLM_BASE_URL")
VLLM_MODEL_NAME: Optional[str] = None  # Discovered at runtime
VLLM_TIMEOUT: int = Field(default=600, env="VLLM_TIMEOUT")
VLLM_MAX_TOKENS: int = 65536
VLLM_TEMPERATURE: float = 0.1
VLLM_RETRY_INTERVAL: int = 30
VLLM_MAX_RETRIES: int = 3
```

**Purpose**: vLLM inference server connection and request parameters

**Environment Variables**:
- `VLLM_BASE_URL`: vLLM API endpoint (default: http://localhost:8078)
- `VLLM_TIMEOUT`: Request timeout in seconds (default: 600)

**Request Parameters** (not configurable via env):
- `MAX_TOKENS`: 65536 (large limit for guided JSON output)
- `TEMPERATURE`: 0.1 (low for consistent extraction)
- `RETRY_INTERVAL`: 30 seconds between retry attempts
- `MAX_RETRIES`: 3 attempts before failing

**Model Discovery**: `VLLM_MODEL_NAME` auto-discovered from /v1/models endpoint

#### GLiNER Configuration

```python
GLINER_MODEL: str = Field(default="urchade/gliner_large-v2.1", env="GLINER_MODEL")
GLINER_THRESHOLD: float = Field(default=0.4, env="GLINER_THRESHOLD")
GLINER_BATCH_SIZE: int = 8
GLINER_MAX_LENGTH: int = 384
```

**Purpose**: GLiNER entity extraction model settings

**Environment Variables**:
- `GLINER_MODEL`: HuggingFace model ID (default: urchade/gliner_large-v2.1)
- `GLINER_THRESHOLD`: Confidence threshold (default: 0.4)

**Model Parameters** (not configurable via env):
- `BATCH_SIZE`: 8 documents per batch
- `MAX_LENGTH`: 384 tokens (GLiNER model limit)

**Model Size**: ~4GB download, ~4GB RAM when loaded

#### Entity Extraction Settings

```python
ENTITY_TAXONOMY_PATH: str = "taxonomy/entities.yaml"
ENTITY_MIN_CONFIDENCE: float = Field(default=0.4, env="GLINER_THRESHOLD")
ENTITY_DEDUPLICATION: bool = True
```

**Purpose**: Entity extraction behavior

**Environment Variables**:
- `GLINER_THRESHOLD`: Used for both GLINER_THRESHOLD and ENTITY_MIN_CONFIDENCE

**Settings** (not configurable via env):
- `TAXONOMY_PATH`: Relative path to entity types YAML
- `DEDUPLICATION`: Enable deduplication (True)

#### Relationship Extraction Settings

```python
RELATION_MIN_CONFIDENCE: float = Field(default=0.45, env="RELATION_MIN_CONFIDENCE")
RELATION_MAX_DISTANCE: int = 3
RELATION_CONTEXT_WINDOW: int = 200
```

**Purpose**: Relationship extraction thresholds

**Environment Variables**:
- `RELATION_MIN_CONFIDENCE`: Minimum confidence for relationships (default: 0.45)

**Settings** (not configurable via env):
- `MAX_DISTANCE`: 3 sentences (unused in current implementation)
- `CONTEXT_WINDOW`: 200 characters of context

#### Co-occurrence Settings

```python
COOCCURRENCE_WINDOW: int = 100
COOCCURRENCE_MIN_COUNT: int = 2
```

**Purpose**: Entity co-occurrence tracking (currently unused)

**Settings** (not configurable via env):
- `WINDOW`: 100 characters
- `MIN_COUNT`: 2 occurrences minimum

#### Processing Settings

```python
MAX_CONCURRENT_REQUESTS: int = 8
REQUEST_TIMEOUT: int = 300
ENABLE_ASYNC_PROCESSING: bool = True
```

**Purpose**: Request handling and concurrency

**Settings** (not configurable via env):
- `MAX_CONCURRENT_REQUESTS`: 8 parallel document processing tasks
- `REQUEST_TIMEOUT`: 300 seconds per request
- `ENABLE_ASYNC_PROCESSING`: True (async/await throughout)

#### Document Processing

```python
MAX_DOCUMENT_LENGTH: int = 100000
CHUNK_SIZE: int = 2000
CHUNK_OVERLAP: int = 200
```

**Purpose**: Document size limits and chunking

**Settings** (not configurable via env):
- `MAX_DOCUMENT_LENGTH`: 100,000 characters max
- `CHUNK_SIZE`: 2000 characters for GLiNER chunking
- `CHUNK_OVERLAP`: 200 characters overlap between chunks

#### Cache Settings

```python
ENABLE_CACHE: bool = True
CACHE_TTL: int = 3600
CACHE_MAX_SIZE: int = 1000
```

**Purpose**: Response caching (currently not implemented)

**Settings** (not configurable via env):
- Future implementation placeholder

#### Rate Limiting

```python
RATE_LIMIT_ENABLED: bool = True
RATE_LIMIT_PER_MINUTE: int = 60
```

**Purpose**: API rate limiting (currently not implemented)

**Settings** (not configurable via env):
- Future implementation placeholder

#### Monitoring

```python
ENABLE_METRICS: bool = True
METRICS_PORT: int = 9090
```

**Purpose**: Prometheus metrics endpoint (currently not implemented)

**Settings** (not configurable via env):
- Future implementation placeholder

### Configuration Loading

**Pydantic Settings Integration**:
```python
class Config:
    env_file = ".env"
    env_file_encoding = "utf-8"
    case_sensitive = True
```

**Loading Priority**:
1. System environment variables
2. .env file in working directory
3. Default values in class definition

**Example .env File**:
```bash
# Service
DEBUG=true
LOG_LEVEL=DEBUG

# API
API_HOST=0.0.0.0
API_PORT=8088

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=secure_password_here

# vLLM
VLLM_BASE_URL=http://localhost:8078
VLLM_TIMEOUT=600

# GLiNER
GLINER_MODEL=urchade/gliner_large-v2.1
GLINER_THRESHOLD=0.4

# Relationships
RELATION_MIN_CONFIDENCE=0.45
```

### Configuration Validation

**Function**: `validate_settings() -> bool`

**Purpose**: Validate critical settings on startup

**Validation Checks**:
1. Neo4j URI is set
2. Neo4j password is set
3. vLLM base URL is set
4. Entity taxonomy file exists at specified path

**Returns**:
- `True`: All validations passed
- `False`: One or more validations failed (errors logged)

**Usage**:
```python
if not validate_settings():
    logger.error("Configuration validation failed!")
    sys.exit(1)
```

**Called By**: `api/server.py` lifespan startup

### Logging Configuration

**Dictionary**: `LOGGING_CONFIG`

**Structure**:
```python
{
  "version": 1,
  "disable_existing_loggers": False,
  "formatters": {
    "default": {
      "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
      "datefmt": "%Y-%m-%d %H:%M:%S"
    },
    "detailed": {
      "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
      "datefmt": "%Y-%m-%d %H:%M:%S"
    }
  },
  "handlers": {
    "console": {
      "class": "logging.StreamHandler",
      "level": settings.LOG_LEVEL,
      "formatter": "default",
      "stream": "ext://sys.stdout"
    },
    "file": {
      "class": "logging.handlers.RotatingFileHandler",
      "level": settings.LOG_LEVEL,
      "formatter": "detailed",
      "filename": "kg-service.log",
      "maxBytes": 10485760,  # 10MB
      "backupCount": 5
    }
  },
  "loggers": {
    "": {
      "level": settings.LOG_LEVEL,
      "handlers": ["console", "file"],
      "propagate": False
    },
    "uvicorn": {
      "level": "INFO",
      "handlers": ["console"],
      "propagate": False
    },
    "fastapi": {
      "level": "INFO",
      "handlers": ["console"],
      "propagate": False
    }
  }
}
```

**Handlers**:
- **Console**: Human-readable format, outputs to stdout
- **File**: Detailed format with function names and line numbers, rotating 10MB files

**Rotation**: 5 backup files retained (50MB total)

**Usage**:
```python
import logging.config
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
```

## vLLM Client

### Purpose

Async HTTP client for vLLM inference server with automatic model discovery, retry logic, and connection management. Handles all LLM inference for relationship extraction.

### Architecture

**Class**: `VLLMClient`

**Dependencies**:
- httpx: Async HTTP client library
- Configuration settings

**Singleton Pattern**: Global instance via `get_vllm_client()`

### Initialization

**Constructor**:
```python
def __init__(self, base_url=None, timeout=None, retry_interval=None):
    self.base_url = base_url or settings.VLLM_BASE_URL
    self.timeout = timeout or settings.VLLM_TIMEOUT
    self.retry_interval = retry_interval or settings.VLLM_RETRY_INTERVAL

    # Model state
    self.model_name: Optional[str] = None  # Discovered via API
    self.last_check: Optional[float] = None
    self.is_available: bool = False

    # HTTP client with connection pooling
    self.client = httpx.AsyncClient(
        timeout=httpx.Timeout(self.timeout),
        limits=httpx.Limits(max_connections=10)
    )
```

**Lifecycle**:
```python
client = await get_vllm_client()  # Get singleton
# ... use client ...
await close_vllm_client()  # Cleanup on shutdown
```

### Model Discovery

**Method**: `get_model_name() -> Optional[str]`

**Purpose**: Query /v1/models endpoint to discover active model

**HTTP Request**:
```
GET {base_url}/v1/models
```

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-3.1-70B-Instruct",
      "object": "model",
      "created": 1234567890,
      "owned_by": "organization"
    }
  ]
}
```

**Returns**: Model ID from first model in list (e.g., "meta-llama/Llama-3.1-70B-Instruct")

**Error Handling**:
- HTTP error: Logs error, returns None
- Empty list: Logs warning, returns None
- Unexpected error: Logs error, returns None

**Method**: `ensure_model() -> bool`

**Purpose**: Ensure model name is discovered before inference

**Logic**:
```python
current_time = time.time()

should_check = (
    self.model_name is None or
    self.last_check is None or
    (current_time - self.last_check) > self.retry_interval
)

if should_check:
    self.model_name = await self.get_model_name()
    self.last_check = current_time
    self.is_available = (self.model_name is not None)

return self.is_available
```

**Retry Behavior**:
- First call: Always checks
- Subsequent calls: Only check if 30+ seconds since last check
- On failure: Will retry after retry_interval seconds

**Method**: `wait_for_model(max_wait_time=300) -> bool`

**Purpose**: Wait for model to become available with periodic retries

**Logic**:
```python
while time_elapsed < max_wait_time:
    if await ensure_model():
        return True
    await asyncio.sleep(retry_interval)
    attempt += 1

return False  # Timeout
```

**Use Case**: Startup waiting for vLLM to load model

### Inference

**Method**: `complete(prompt, max_tokens=None, temperature=None, stop=None, **kwargs) -> str`

**Purpose**: Generate text completion from vLLM

**HTTP Request**:
```
POST {base_url}/v1/completions
Content-Type: application/json

{
  "model": "meta-llama/Llama-3.1-70B-Instruct",
  "prompt": "Extract relationships from...",
  "max_tokens": 65536,
  "temperature": 0.1,
  "stop": ["```", "\n\n\n"]
}
```

**Response**:
```json
{
  "id": "cmpl-123",
  "object": "text_completion",
  "created": 1234567890,
  "model": "meta-llama/Llama-3.1-70B-Instruct",
  "choices": [
    {
      "text": "[{\"subject\": \"FastAPI\", \"predicate\": \"uses\", ...}]",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ]
}
```

**Returns**: Generated text from choices[0].text

**Error Handling**:
- Model unavailable: Raises `ModelUnavailableError`
- HTTP error: Logs error, resets model state, raises `ModelUnavailableError`
- Unexpected error: Logs error, resets model state, raises exception

**State Reset on Failure**:
```python
def reset_model_state(self):
    self.model_name = None
    self.last_check = None
    self.is_available = False
```

Triggers model rediscovery on next request

**Method**: `extract_json(prompt, max_tokens=None, temperature=None) -> Dict`

**Purpose**: Generate JSON completion and parse response

**Process**:
1. Call `complete()` with stop sequences ["```", "\n\n\n"]
2. Try direct JSON parsing
3. If fails, extract JSON from markdown code blocks
4. If fails, find any JSON object in response
5. If all fail, raise ValueError

**Returns**: Parsed JSON dictionary

**Error Handling**:
- JSON parse failure: Raises ValueError with response preview

### Health Check

**Method**: `health_check() -> bool`

**Purpose**: Check if vLLM server is responsive

**HTTP Request**:
```
GET {base_url}/health
```

**Returns**: True if status 200, False otherwise

**Use Case**: Health endpoint at API layer

**Method**: `get_model_info() -> Optional[Dict]`

**Purpose**: Get detailed model information

**Returns**: First model object from /v1/models endpoint

**Use Case**: Model info endpoint at API layer

### Connection Management

**Context Manager Support**:
```python
async with VLLMClient() as client:
    response = await client.complete("...")
    # Automatically closes on exit
```

**Explicit Lifecycle**:
```python
client = VLLMClient()
# ... operations ...
await client.close()
```

**Connection Pooling**:
- httpx client maintains pool of 10 connections
- Reused across requests
- Automatic keepalive

### Retry Logic

**Exponential Backoff** (not currently implemented, fixed interval):
- Retry interval: 30 seconds
- Max retries: 3 attempts
- Reset model state on each failure

**Use Case**: Transient vLLM failures (server restart, model loading)

### Error Types

**ModelUnavailableError**:
```python
class ModelUnavailableError(Exception):
    """Raised when vLLM model is not available"""
```

**Raised When**:
- `ensure_model()` returns False
- HTTP request fails
- Timeout exceeded

**Handled By**: RelationshipExtractor (logs warning, returns empty list)

## Environment Setup

### Development Environment

**.env File**:
```bash
DEBUG=true
LOG_LEVEL=DEBUG
API_HOST=0.0.0.0
API_PORT=8088
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=dev_password
VLLM_BASE_URL=http://localhost:8078
```

**Start Services**:
```bash
# Neo4j
docker compose up -d neo4j

# vLLM (external)
# Run vLLM server separately on port 8078

# kg-service
python main.py
```

### Production Environment

**Environment Variables** (container deployment):
```bash
DEBUG=false
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8088
NEO4J_URI=bolt://neo4j-cluster:7687
NEO4J_PASSWORD=<secure-password>
VLLM_BASE_URL=http://vllm-service:8078
VLLM_TIMEOUT=600
GLINER_THRESHOLD=0.4
RELATION_MIN_CONFIDENCE=0.45
```

**Security Recommendations**:
1. Always override `NEO4J_PASSWORD` in production
2. Use TLS for Neo4j connections (bolt+s://)
3. Keep vLLM on internal network (no public exposure)
4. Configure API gateway for authentication (kg-service has none)
5. Restrict API_HOST to specific interface if needed

### Testing Environment

**Override Settings**:
```python
from config import Settings

test_settings = Settings(
    DEBUG=True,
    NEO4J_URI="bolt://localhost:7687",
    NEO4J_PASSWORD="test",
    VLLM_BASE_URL="http://localhost:8078"
)
```

**Mock vLLM**:
```python
class MockVLLMClient:
    async def complete(self, prompt, **kwargs):
        return '[{"subject": "test", "predicate": "uses", "object": "mock"}]'
```

## Configuration Best Practices

### Security
- Never commit .env file to version control
- Use environment-specific .env files (.env.dev, .env.prod)
- Rotate passwords regularly
- Use secrets management (AWS Secrets Manager, HashiCorp Vault) in production

### Performance
- Increase `NEO4J_MAX_CONNECTION_POOL_SIZE` for high concurrency
- Tune `VLLM_TIMEOUT` based on model size
- Adjust `MAX_CONCURRENT_REQUESTS` based on available resources
- Lower `GLINER_THRESHOLD` for higher recall (more entities)
- Raise `RELATION_MIN_CONFIDENCE` for higher precision (fewer false relationships)

### Logging
- Use `LOG_LEVEL=INFO` in production
- Use `LOG_LEVEL=DEBUG` for troubleshooting
- Monitor log file growth (10MB files with 5 backups = 50MB max)
- Consider centralized logging (ELK, Splunk) in production

### Monitoring
- Implement Prometheus metrics (placeholder exists)
- Monitor vLLM availability and latency
- Track Neo4j connection pool utilization
- Alert on processing failures

---

[Back to Home](index.md)
