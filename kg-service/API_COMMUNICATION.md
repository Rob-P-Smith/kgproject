# API Communication Guide

## Overview

This document describes the API communication between **mcpragcrawl4ai** and **kg-service**.

## Architecture

```
┌─────────────────────────┐         HTTP POST          ┌─────────────────────────┐
│   mcpragcrawl4ai        │    /api/v1/ingest          │     kg-service          │
│   (crawl4ai-rag-server) │ ───────────────────────►  │   (kg-service:8088)     │
│                         │                            │                         │
│  - Crawls documents     │                            │  - Extract entities     │
│  - Cleans markdown      │                            │  - Extract relationships│
│  - Chunks content       │                            │  - Map to chunks        │
│  - Generates embeddings │                            │  - Store in Neo4j       │
│  - Queues for KG        │                            │  - Return results       │
│                         │   ◄───────────────────────  │                         │
│  - Stores results       │      JSON Response         │                         │
└─────────────────────────┘                            └─────────────────────────┘
```

## Endpoints

### 1. Health Check

**GET** `/health`

Check if kg-service is healthy.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-15T12:00:00Z",
  "services": {
    "neo4j": "connected",
    "vllm": "connected (Qwen2.5-7B-Instruct)",
    "gliner": "loaded"
  },
  "version": "1.0.0",
  "uptime_seconds": 3600.5
}
```

---

### 2. Service Statistics

**GET** `/stats`

Get processing statistics.

**Response:**
```json
{
  "total_documents_processed": 523,
  "total_entities_extracted": 45234,
  "total_relationships_extracted": 12456,
  "avg_processing_time_ms": 2341.5,
  "last_processed_at": "2025-10-15T12:00:00Z",
  "queue_size": 0,
  "failed_count": 2
}
```

---

### 3. Ingest Document (Main Endpoint)

**POST** `/api/v1/ingest`

Process document for entity/relationship extraction.

**Request Body:**
```json
{
  "content_id": 123,
  "url": "https://docs.fastapi.com",
  "title": "FastAPI Documentation",
  "markdown": "# FastAPI\n\nFastAPI is a modern web framework for building APIs with Python 3.7+ based on standard Python type hints...",
  "chunks": [
    {
      "vector_rowid": 45001,
      "chunk_index": 0,
      "char_start": 0,
      "char_end": 2500,
      "text": "# FastAPI\n\nFastAPI is a modern web framework..."
    },
    {
      "vector_rowid": 45002,
      "chunk_index": 1,
      "char_start": 2450,
      "char_end": 4950,
      "text": "...framework for building APIs with Python 3.7+..."
    }
  ],
  "metadata": {
    "tags": "python,web,api",
    "timestamp": "2025-10-15T12:00:00Z"
  }
}
```

**Response:** (200 OK)
```json
{
  "success": true,
  "content_id": 123,
  "neo4j_document_id": "4:doc:456",
  "entities_extracted": 87,
  "relationships_extracted": 43,
  "processing_time_ms": 2341,
  "entities": [
    {
      "text": "FastAPI",
      "normalized": "fastapi",
      "type_primary": "Framework",
      "type_sub1": "Backend",
      "type_sub2": "Python",
      "type_sub3": null,
      "type_full": "Framework::Backend::Python",
      "confidence": 0.95,
      "neo4j_node_id": "4:entity:789",
      "context_before": "modern web ",
      "context_after": " for building",
      "sentence": "FastAPI is a modern web framework for building APIs.",
      "chunk_appearances": [
        {
          "vector_rowid": 45001,
          "chunk_index": 0,
          "offset_start": 342,
          "offset_end": 349
        },
        {
          "vector_rowid": 45002,
          "chunk_index": 1,
          "offset_start": 73,
          "offset_end": 80
        }
      ],
      "spans_multiple_chunks": true
    }
  ],
  "relationships": [
    {
      "subject_text": "FastAPI",
      "subject_neo4j_id": "4:entity:789",
      "predicate": "uses",
      "object_text": "Pydantic",
      "object_neo4j_id": "4:entity:790",
      "confidence": 0.88,
      "context": "FastAPI uses Pydantic for data validation",
      "neo4j_relationship_id": "5:rel:101",
      "spans_chunks": false,
      "chunk_rowids": [45001]
    }
  ],
  "summary": {
    "entities_by_type": {
      "Framework": 12,
      "Language": 3,
      "Concept": 5
    },
    "relationships_by_predicate": {
      "uses": 15,
      "competes_with": 3,
      "implements": 8
    },
    "chunks_with_entities": 18,
    "avg_entities_per_chunk": 4.8
  }
}
```

**Error Response:** (500 Internal Server Error)
```json
{
  "success": false,
  "error": "Failed to extract entities: GLiNER model not loaded",
  "error_type": "ProcessingError",
  "content_id": 123,
  "timestamp": "2025-10-15T12:00:00Z"
}
```

---

### 4. Model Information

**GET** `/api/v1/model-info`

Get information about loaded models.

**Response:**
```json
{
  "gliner": {
    "model": "urchade/gliner_large-v2.1",
    "threshold": 0.5,
    "status": "loaded",
    "entity_types_count": 302
  },
  "vllm": {
    "base_url": "http://host.docker.internal:8078",
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "status": "connected"
  }
}
```

---

## Using the Client

### Python Client (for mcpragcrawl4ai)

Place `kg-service-client.py` in `mcpragcrawl4ai/core/clients/kg_service_client.py`.

**Example Usage:**

```python
from core.clients.kg_service_client import KGServiceClient, KGServiceError

async def process_with_kg(content_id, url, title, markdown, chunks):
    """Process document with KG service"""

    async with KGServiceClient("http://kg-service:8088") as kg_client:
        try:
            # Check health first
            health = await kg_client.health_check()
            print(f"KG Service: {health['status']}")

            # Send for processing
            result = await kg_client.ingest_document(
                content_id=content_id,
                url=url,
                title=title,
                markdown=markdown,
                chunks=chunks,
                metadata={
                    "tags": "python,web",
                    "timestamp": datetime.now().isoformat()
                }
            )

            # Result contains entities and relationships
            print(f"Entities: {result['entities_extracted']}")
            print(f"Relationships: {result['relationships_extracted']}")

            return result

        except KGServiceError as e:
            print(f"KG processing failed: {e}")
            return None
```

**Safe Mode (non-blocking):**

```python
# Use this if KG failures should not block the main pipeline
result = await kg_client.ingest_document_safe(
    content_id=content_id,
    url=url,
    title=title,
    markdown=markdown,
    chunks=chunks
)

if result:
    print("KG processing successful")
else:
    print("KG processing failed, continuing without graph data")
```

---

## Request Validation

The API validates:

1. **Required Fields:**
   - `content_id` (must be > 0)
   - `url` (must start with http:// or https://)
   - `title` (max 500 chars)
   - `markdown` (50 - 1,000,000 chars)
   - `chunks` (min 1, max 1000)

2. **Chunk Validation:**
   - Must be ordered by `chunk_index`
   - `char_end` > `char_start`
   - All fields required

3. **Field Limits:**
   - URL: 2048 chars max
   - Title: 500 chars max
   - Markdown: 1,000,000 chars max
   - Chunks: 1000 max

---

## Error Handling

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 422 | Validation error (invalid request) |
| 500 | Processing error |
| 503 | Service unavailable |

### Error Types

1. **Validation Errors (422):**
   - Missing required fields
   - Invalid data types
   - Chunk ordering issues
   - Field length violations

2. **Processing Errors (500):**
   - GLiNER model failure
   - vLLM timeout or error
   - Neo4j connection failure
   - Entity/relationship extraction failure

3. **Service Unavailable (503):**
   - KG processor not initialized
   - Neo4j not connected
   - Critical service failure

---

## Timeouts

| Operation | Default Timeout | Configurable Via |
|-----------|-----------------|------------------|
| HTTP Request | 300s (5 min) | Client initialization |
| Entity Extraction | No limit | GLiNER processes in-memory |
| Relationship Extraction | 120s | `VLLM_TIMEOUT` env var |
| Neo4j Storage | 30s | `NEO4J_CONNECTION_TIMEOUT` |

---

## Performance Considerations

### Request Size

- **Small document** (< 5KB markdown): ~1-2 seconds
- **Medium document** (5-50KB): ~2-10 seconds
- **Large document** (50-200KB): ~10-30 seconds
- **Very large** (> 200KB): ~30+ seconds

### Concurrent Requests

kg-service can handle multiple concurrent requests:
- Default: 10 concurrent connections
- Configurable via `MAX_CONCURRENT_REQUESTS`

### Retry Logic

Client automatically retries on:
- Connection errors (up to 3 times with exponential backoff)
- HTTP 5xx errors (up to 3 times)
- Does NOT retry on 4xx errors (validation issues)

---

## Testing

### Local Testing (without Docker)

```bash
# Start kg-service locally
cd kg-service
python main.py

# In another terminal, test with curl
curl http://localhost:8088/health

curl -X POST http://localhost:8088/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d @test_data/sample_request.json
```

### Docker Testing

```bash
# Start services
docker compose up -d

# Check kg-service logs
docker compose logs -f kg-service

# Test from mcpragcrawl4ai container
docker exec -it crawl4ai-rag-server python3 -c "
import asyncio
from core.clients.kg_service_client import KGServiceClient

async def test():
    async with KGServiceClient('http://kg-service:8088') as client:
        health = await client.health_check()
        print(health)

asyncio.run(test())
"
```

### Unit Tests

```bash
cd kg-service
pytest tests/test_api.py -v
```

---

## Monitoring

### Metrics to Monitor

1. **Request Rate:**
   - `/stats` endpoint shows `total_documents_processed`

2. **Processing Time:**
   - Average: `avg_processing_time_ms`
   - Per-request: `processing_time_ms` in response

3. **Error Rate:**
   - Failed count: `/stats` → `failed_count`

4. **Service Health:**
   - `/health` endpoint status
   - Dependent service status (neo4j, vllm, gliner)

### Logging

All requests are logged with:
- Method and path
- Status code
- Processing time
- Content ID (for ingest requests)
- Error details (if failed)

Log format:
```
INFO: POST /api/v1/ingest - Status: 200 - Time: 2341.52ms
INFO: Processing document: https://docs.fastapi.com
INFO: ✓ Processing complete: Entities: 87, Relationships: 43
```

---

## Next Steps

1. **Integration:** Add KG client to mcpragcrawl4ai queue worker
2. **Testing:** Test full pipeline end-to-end
3. **Monitoring:** Set up metrics collection
4. **Optimization:** Tune timeout and retry settings based on usage

---

**Status:** API communication layer complete and ready for integration.
