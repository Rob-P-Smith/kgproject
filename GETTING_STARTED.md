# Getting Started with Knowledge Graph RAG System

Complete guide to set up and run the Knowledge Graph enhanced RAG system.

---

## Prerequisites

### Required:
- **Docker** and **Docker Compose** installed
- **8GB RAM minimum** (16GB recommended)
- **20GB disk space** for models and data
- **vLLM server** running on host at `localhost:8078` (optional but recommended)

### Optional:
- Python 3.11+ for local development
- Neo4j Browser for graph visualization
- curl or httpx for API testing

---

## Quick Start (5 minutes)

### 1. Clone and Navigate

```bash
cd /home/robiloo/Documents/KG-project
```

### 2. Configure Environment

```bash
# Review and edit .env file
nano .env

# Key settings:
# - NEO4J_PASSWORD (change from default!)
# - VLLM_BASE_URL (if vLLM on different port)
# - KG_LOG_LEVEL (INFO for normal, DEBUG for troubleshooting)
```

### 3. Start Services

```bash
# Start Neo4j first (waits for health check)
docker-compose up -d neo4j

# Wait 30 seconds for Neo4j to be healthy
docker-compose ps

# Start kg-service (depends on neo4j)
docker-compose up -d kg-service

# Check logs
docker-compose logs -f kg-service
```

### 4. Verify Services

```bash
# Check Neo4j
curl http://localhost:7474
# Expected: Neo4j Browser interface

# Check kg-service health
curl http://localhost:8088/health
# Expected: {"status":"healthy",...}

# Check service info
curl http://localhost:8088/api/v1/model-info
```

### 5. Test with Sample Document

```bash
# Use the test client
cd kg-service
python kg-service-client.py test
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     mcpragcrawl4ai                           │
│  Port: 8080 (crawl4ai-rag-server container)                 │
│  - Crawls web pages                                          │
│  - Chunks content (500 words, 50 overlap)                    │
│  - Generates embeddings (all-MiniLM-L6-v2)                   │
│  - Stores in SQLite + sqlite-vec                             │
│  - Tracks chunk boundaries                                   │
│  - Queues for KG processing (if service available)           │
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTP POST /api/v1/ingest
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                      kg-service                              │
│  Port: 8088 (kg-service container)                          │
│  - Receives full markdown + chunk metadata                   │
│  - Extracts entities with GLiNER (300+ types)                │
│  - Extracts relationships with vLLM                          │
│  - Maps to chunk boundaries                                  │
│  - Stores in Neo4j                                           │
│  - Returns entities/relationships to mcpragcrawl4ai          │
└──────────────────────┬───────────────────────────────────────┘
                       │ Cypher queries
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                       Neo4j                                  │
│  Ports: 7474 (HTTP), 7687 (Bolt)                            │
│  - Stores Document/Chunk/Entity nodes                        │
│  - Stores relationships (USES, IMPLEMENTS, etc.)             │
│  - Enables graph traversal                                   │
│  - Co-occurrence tracking                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Setup

### Step 1: Neo4j Configuration

**Memory Settings** (in `.env`):

```bash
# For 16GB system:
NEO4J_HEAP_INITIAL=512m
NEO4J_HEAP_MAX=4G
NEO4J_PAGECACHE=8G

# For 8GB system:
NEO4J_HEAP_INITIAL=512m
NEO4J_HEAP_MAX=2G
NEO4J_PAGECACHE=2G
```

**Security:**

```bash
# Change default password!
NEO4J_PASSWORD=your_secure_password_here
```

**Verify Neo4j:**

```bash
# Check if running
docker-compose ps neo4j

# Check logs
docker-compose logs neo4j

# Access browser
open http://localhost:7474

# Login with:
# Username: neo4j
# Password: knowledge_graph_2024 (or your custom password)
```

### Step 2: vLLM Setup (Optional but Recommended)

**If you don't have vLLM running**, relationship extraction will be skipped but entity extraction will still work.

**To start vLLM:**

```bash
# Example: Start vLLM with Qwen2.5-7B
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8078 \
  --max-model-len 4096

# Or use Docker:
docker run --gpus all \
  -p 8078:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct
```

**Verify vLLM:**

```bash
curl http://localhost:8078/v1/models
# Expected: {"data":[{"id":"Qwen/Qwen2.5-7B-Instruct",...}]}
```

### Step 3: kg-service Configuration

**Environment Variables** (in `.env`):

```bash
# Debug mode (set to true for verbose logging)
KG_DEBUG=false

# Log level (DEBUG, INFO, WARNING, ERROR)
KG_LOG_LEVEL=INFO

# vLLM endpoint (adjust if different port)
VLLM_BASE_URL=http://host.docker.internal:8078

# GLiNER model (default is good)
GLINER_MODEL=urchade/gliner_large-v2.1
```

**First Run** (downloads models):

```bash
# Start kg-service
docker-compose up kg-service

# Watch logs (model download takes 5-10 minutes)
docker-compose logs -f kg-service

# Expected output:
# "Downloading GLiNER model..."
# "✓ GLiNER model loaded"
# "Checking vLLM availability..."
# "✓ vLLM ready: Qwen/Qwen2.5-7B-Instruct"
# "✓ kg-service ready"
```

**Models are cached** in `kg-models` Docker volume for fast restarts.

### Step 4: mcpragcrawl4ai Integration

**Update mcpragcrawl4ai** to enable KG integration:

```bash
cd /home/robiloo/Documents/mcpragcrawl4ai

# Set environment variables (in docker-compose.yml or .env)
KG_SERVICE_ENABLED=true
KG_SERVICE_URL=http://kg-service:8088
KG_HEALTH_CHECK_INTERVAL=30.0
```

**Restart mcpragcrawl4ai:**

```bash
docker-compose restart crawl4ai-rag-server
```

**Check integration:**

```bash
# Check mcpragcrawl4ai logs
docker-compose logs -f crawl4ai-rag-server

# Expected:
# "✓ KG service is now healthy (status: healthy)"
# "KG Service Config: enabled=True, url=http://kg-service:8088"
```

---

## Usage Examples

### Example 1: Crawl and Process Document

**From mcpragcrawl4ai:**

```bash
curl -X POST http://localhost:8080/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://docs.python.org/3/tutorial/",
    "tags": "python,tutorial"
  }'
```

**What happens:**
1. mcpragcrawl4ai crawls URL
2. Cleans content (removes navigation)
3. Chunks into 500-word pieces
4. Generates embeddings
5. Stores in SQLite
6. **Checks KG service health**
7. **If healthy:** Queues for KG processing
8. **If unhealthy:** Marks as skipped (can retry later)

**Check queue status:**

```bash
sqlite3 /app/data/crawl4ai_rag.db \
  "SELECT content_id, status, skipped_reason FROM kg_processing_queue ORDER BY id DESC LIMIT 5;"
```

### Example 2: Direct KG Service Test

**Send document directly to kg-service:**

```bash
curl -X POST http://localhost:8088/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content_id": 999,
    "url": "https://example.com/test",
    "title": "Test Document",
    "markdown": "# FastAPI\n\nFastAPI is a modern Python web framework. It uses Pydantic for data validation.",
    "chunks": [
      {
        "vector_rowid": 1,
        "chunk_index": 0,
        "char_start": 0,
        "char_end": 100,
        "text": "FastAPI is a modern Python web framework. It uses Pydantic for data validation."
      }
    ],
    "metadata": {"tags": "python,web"}
  }'
```

**Expected response:**

```json
{
  "success": true,
  "content_id": 999,
  "neo4j_document_id": "4:abc123:456",
  "entities_extracted": 3,
  "relationships_extracted": 1,
  "entities": [
    {
      "text": "FastAPI",
      "type_primary": "Framework",
      "type_sub1": "Backend",
      "type_sub2": "Python",
      "confidence": 0.95,
      "chunk_appearances": [{"vector_rowid": 1, ...}]
    },
    {
      "text": "Python",
      "type_primary": "Language",
      "type_sub1": "Interpreted",
      "confidence": 0.92,
      "chunk_appearances": [{"vector_rowid": 1, ...}]
    },
    {
      "text": "Pydantic",
      "type_primary": "Framework",
      "type_sub1": "Data",
      "confidence": 0.88,
      "chunk_appearances": [{"vector_rowid": 1, ...}]
    }
  ],
  "relationships": [
    {
      "subject_text": "FastAPI",
      "predicate": "uses",
      "object_text": "Pydantic",
      "confidence": 0.85,
      "context": "It uses Pydantic for data validation"
    }
  ]
}
```

### Example 3: Query Neo4j Graph

**From Neo4j Browser** (http://localhost:7474):

```cypher
// Find all entities of type Framework
MATCH (e:Entity {type_primary: "Framework"})
RETURN e.text, e.type_sub1, e.mention_count
ORDER BY e.mention_count DESC
LIMIT 10;

// Find relationships between entities
MATCH (e1:Entity)-[r:USES]->(e2:Entity)
RETURN e1.text, type(r), e2.text, r.confidence
ORDER BY r.confidence DESC
LIMIT 10;

// Find entities that co-occur frequently
MATCH (e1:Entity)-[co:CO_OCCURS_WITH]->(e2:Entity)
WHERE co.count >= 3
RETURN e1.text, e2.text, co.count
ORDER BY co.count DESC
LIMIT 10;

// Find all chunks for a document
MATCH (d:Document {content_id: 123})-[:HAS_CHUNK]->(c:Chunk)
RETURN c.chunk_index, c.char_start, c.char_end, c.text_preview
ORDER BY c.chunk_index;
```

---

## Monitoring and Troubleshooting

### Health Checks

```bash
# Check all services
docker-compose ps

# kg-service health
curl http://localhost:8088/health
# Expected: {"status":"healthy","services":{"neo4j":"connected","vllm":"connected","gliner":"loaded"}}

# Service statistics
curl http://localhost:8088/stats
# Expected: {"total_documents_processed":10,"total_entities_extracted":250,...}
```

### Common Issues

#### Issue 1: kg-service shows "unhealthy"

**Check logs:**
```bash
docker-compose logs kg-service | tail -50
```

**Common causes:**
- Neo4j not ready → Wait 30s after starting Neo4j
- vLLM not accessible → Check `VLLM_BASE_URL` in `.env`
- GLiNER model download failed → Check disk space

**Solution:**
```bash
# Restart kg-service
docker-compose restart kg-service

# Watch logs
docker-compose logs -f kg-service
```

#### Issue 2: "KG service unavailable" in mcpragcrawl4ai

**This is expected** if kg-service is down or starting up.

**Behavior:**
- Content still gets crawled and stored
- Chunks tracked with `kg_processed=0`
- Queue entries marked as `skipped`
- RAG search works with vector-only (graceful fallback)

**Check mcpragcrawl4ai database:**
```bash
sqlite3 /app/data/crawl4ai_rag.db \
  "SELECT COUNT(*) FROM kg_processing_queue WHERE status='skipped' AND skipped_reason='kg_service_unavailable';"
```

**Retry later:**
Background worker (when implemented) will retry skipped items automatically.

#### Issue 3: vLLM connection errors

**Symptoms:**
- Entities extracted but no relationships
- Logs show "vLLM not immediately available"

**This is OK** - entity extraction still works without vLLM.

**To fix:**
```bash
# Check vLLM is running
curl http://localhost:8078/v1/models

# If not, start vLLM (see Step 2)

# kg-service will auto-reconnect on next health check (30s interval)
```

#### Issue 4: Out of memory errors

**Symptoms:**
- Neo4j crashes
- kg-service killed by OOM
- Docker compose stops unexpectedly

**Solution:**
```bash
# Reduce memory in .env
NEO4J_HEAP_MAX=2G
NEO4J_PAGECACHE=2G

# Restart services
docker-compose down
docker-compose up -d
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f kg-service
docker-compose logs -f neo4j

# Last 100 lines
docker-compose logs --tail=100 kg-service

# Since timestamp
docker-compose logs --since="2024-01-01T12:00:00" kg-service
```

---

## Performance Tuning

### GLiNER (Entity Extraction)

**Faster but less accurate:**
```bash
# In .env
GLINER_MODEL=urchade/gliner_small-v2.1
GLINER_THRESHOLD=0.6
```

**Slower but more accurate:**
```bash
GLINER_MODEL=urchade/gliner_large-v2.1
GLINER_THRESHOLD=0.4
```

### Neo4j Memory

**For 32GB system:**
```bash
NEO4J_HEAP_MAX=8G
NEO4J_PAGECACHE=16G
```

**For 8GB system:**
```bash
NEO4J_HEAP_MAX=2G
NEO4J_PAGECACHE=2G
```

### Batch Processing

**Process multiple documents:**

```python
import asyncio
from kg_service_client import KGServiceClient

async def batch_process(urls):
    client = KGServiceClient()

    tasks = []
    for url in urls:
        # Crawl with mcpragcrawl4ai first
        # Then get chunk metadata
        # Then send to kg-service
        task = client.ingest_document_safe(...)
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results
```

---

## Next Steps

1. **Implement Background Worker** - Process kg_processing_queue automatically
2. **Graph-Enhanced Search** - Use entity/relationship data in search
3. **Query API** - Add endpoints for graph traversal queries
4. **Monitoring Dashboard** - Visualize processing stats
5. **Retry Logic** - Automatically retry failed/skipped items

---

## Support and Documentation

- **KG Service API**: [API_COMMUNICATION.md](kg-service/API_COMMUNICATION.md)
- **Integration Plan**: [KGPlan.md](KGPlan.md) and [RetrievalPlan.md](RetrievalPlan.md)
- **Content Cleaning**: [mcpragcrawl4ai/CONTENT_CLEANING_PIPELINE.md](../mcpragcrawl4ai/CONTENT_CLEANING_PIPELINE.md)
- **KG Integration**: [mcpragcrawl4ai/KG_INTEGRATION.md](../mcpragcrawl4ai/KG_INTEGRATION.md)

---

## Quick Reference

**Start everything:**
```bash
docker-compose up -d
```

**Stop everything:**
```bash
docker-compose down
```

**Restart kg-service:**
```bash
docker-compose restart kg-service
```

**View logs:**
```bash
docker-compose logs -f kg-service
```

**Check health:**
```bash
curl http://localhost:8088/health
```

**Neo4j Browser:**
```
http://localhost:7474
Username: neo4j
Password: knowledge_graph_2024
```

---

**You're ready to start using the Knowledge Graph RAG system!** 🚀
