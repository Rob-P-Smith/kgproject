# Knowledge Graph System - Implementation Plan (Part 1: Data Input & Processing)

**Document Version:** 1.0
**Created:** 2025-10-15
**Status:** Planning Phase

---

## Table of Contents - Part 1

1. [System Overview](#1-system-overview)
2. [Architecture Components](#2-architecture-components)
3. [Data Flow: Ingestion Pipeline](#3-data-flow-ingestion-pipeline)
4. [Database Schema Design](#4-database-schema-design)
5. [Entity Extraction System](#5-entity-extraction-system)
6. [Relationship Extraction System](#6-relationship-extraction-system)
7. [Chunk Mapping System](#7-chunk-mapping-system)
8. [Neo4j Graph Storage](#8-neo4j-graph-storage)
9. [SQLite Write-back System](#9-sqlite-write-back-system)

---

## 1. System Overview

### 1.1 Goal
Build an enterprise-level Knowledge Graph enhanced RAG system that:
- Extracts 300+ hierarchical entity types from documents
- Identifies relationships between entities using LLM reasoning
- Maps entities/relationships to document chunks for precise retrieval
- Enables multi-hop graph traversal for hidden knowledge discovery
- Provides configurable exploration depth for query expansion

### 1.2 Core Principle
**Process full documents for context, map to chunks for retrieval.**

### 1.3 System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     mcpragcrawl4ai                               │
│  - Web crawling (Crawl4AI)                                       │
│  - Content cleaning                                              │
│  - Document chunking                                             │
│  - Vector embeddings (SentenceTransformers)                      │
│  - SQLite storage + sqlite-vec                                   │
│  - Queue management for KG processing                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      kg-service                                  │
│  - Entity extraction (GLiNER)                                    │
│  - Relationship extraction (vLLM)                                │
│  - Chunk mapping                                                 │
│  - Neo4j storage                                                 │
│  - FastAPI endpoints                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Neo4j                                    │
│  - Graph database                                                │
│  - Document/Chunk/Entity nodes                                   │
│  - Relationship storage                                          │
│  - Graph traversal queries                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture Components

### 2.1 Docker Containers

```yaml
# Docker Network: crawler_default
containers:
  - name: crawl4ai-rag-server
    purpose: RAG system with vector search
    ports:
      - 8080 (REST API)
      - 3000 (MCP Server)

  - name: neo4j-kg
    purpose: Graph database
    ports:
      - 7474 (HTTP Browser)
      - 7687 (Bolt Protocol)

  - name: kg-service
    purpose: Entity/relationship processing
    ports:
      - 8088 (REST API)

  - name: vllm-server (external)
    purpose: LLM inference for relationships
    ports:
      - 8078 (OpenAI-compatible API)
```

### 2.2 Technology Stack

**mcpragcrawl4ai:**
- Python 3.11+
- SQLite + sqlite-vec
- SentenceTransformers (all-MiniLM-L6-v2)
- Crawl4AI for web scraping
- FastAPI for REST API

**kg-service:**
- Python 3.11+
- GLiNER (urchade/gliner_large-v2.1)
- httpx/aiohttp for vLLM client
- neo4j Python driver
- FastAPI for REST API

**Neo4j:**
- Neo4j Community Edition 5.25
- APOC plugin for procedures
- Native graph storage

**vLLM Server (external):**
- Running on host at localhost:8078
- Serves LLM for relationship extraction
- Auto-discovery via /v1/models endpoint

---

## 3. Data Flow: Ingestion Pipeline

### 3.1 Overview

```
User Request → Crawl → Clean → Chunk → Embed → Store SQLite
                                                     ↓
                                            Queue for KG Processing
                                                     ↓
                                          Background Worker (async)
                                                     ↓
                                        POST to kg-service:8088
                                        {full_markdown, chunks[]}
                                                     ↓
                                            Entity Extraction (GLiNER)
                                                     ↓
                                        Relationship Extraction (vLLM)
                                                     ↓
                                            Map to Chunks (positions)
                                                     ↓
                                              Store in Neo4j
                                                     ↓
                                        Return: entities, relationships,
                                                chunk_mappings
                                                     ↓
                                    mcpragcrawl4ai writes back to SQLite
                                            (chunk_entities table)
```

### 3.2 Step-by-Step Flow

#### Step 1: Crawl & Clean (mcpragcrawl4ai)

**Input:**
- User request: `crawl_and_store("https://docs.fastapi.com", tags="python,api")`

**Process:**
1. Crawl4AI fetches URL
2. Convert HTML to markdown (fit_markdown)
3. ContentCleaner removes navigation, boilerplate
4. Language detection (skip non-English)
5. Calculate content hash (SHA256)

**Output:**
- `cleaned_markdown` (full document text)
- `url`, `title`, `metadata`

**Code Location:** `mcpragcrawl4ai/core/operations/crawler.py::crawl_and_store()`

---

#### Step 2: Chunk & Embed (mcpragcrawl4ai)

**Input:**
- `cleaned_markdown` (full document)

**Process:**
1. Split into chunks:
   ```python
   chunk_size = 500 words
   overlap = 50 words
   chunks = chunk_content(cleaned_markdown, chunk_size, overlap)
   ```

2. For each chunk:
   ```python
   embedding = model.encode(chunk_text)  # 384-dim vector
   vector_rowid = insert_into_content_vectors(embedding, content_id)

   # NEW: Track chunk metadata
   insert_into_content_chunks(
       rowid=vector_rowid,
       content_id=content_id,
       chunk_index=i,
       chunk_text=chunk_text,
       char_start=calculate_position_in_original(i),
       char_end=calculate_position_in_original(i) + len(chunk_text)
   )
   ```

**Output:**
- Vector embeddings stored in `content_vectors` virtual table
- Chunk metadata stored in new `content_chunks` table
- `chunk_metadata[]` array for KG processing

**Code Location:** `mcpragcrawl4ai/core/data/storage.py::store_content()`

---

#### Step 3: Queue for KG Processing (mcpragcrawl4ai)

**Input:**
- `content_id` (SQLite row ID)
- `cleaned_markdown` (full document)
- `chunk_metadata[]` (chunk positions and vector_rowids)

**Process:**
1. Check if KG processing enabled:
   ```python
   if not settings.KG_SERVICE_ENABLED:
       return  # Skip KG processing
   ```

2. Insert into queue table:
   ```sql
   INSERT INTO kg_processing_queue (
       content_id,
       status,
       queued_at,
       priority
   ) VALUES (?, 'pending', CURRENT_TIMESTAMP, 1);
   ```

3. Queue entry includes:
   ```python
   {
       "content_id": 123,
       "url": "https://docs.fastapi.com",
       "title": "FastAPI Documentation",
       "markdown": "full cleaned markdown...",
       "chunks": [
           {
               "vector_rowid": 45001,
               "chunk_index": 0,
               "char_start": 0,
               "char_end": 2500,
               "text": "first 500 words..."
           },
           {
               "vector_rowid": 45002,
               "chunk_index": 1,
               "char_start": 2450,  # 50-word overlap
               "char_end": 4950,
               "text": "next 500 words..."
           },
           // ... all chunks
       ],
       "metadata": {
           "tags": "python,api",
           "timestamp": "2025-10-15T12:00:00Z",
           "word_count": 10234
       }
   }
   ```

**Output:**
- Queue table row with status='pending'
- Full document + chunk boundaries ready for processing

**Code Location:**
- `mcpragcrawl4ai/core/data/storage.py::queue_for_kg_processing()`
- NEW: `mcpragcrawl4ai/core/data/kg_queue.py` (queue manager)

---

#### Step 4: Background Worker (mcpragcrawl4ai)

**Purpose:** Process queue asynchronously without blocking crawls

**Implementation:**

```python
# mcpragcrawl4ai/core/workers/kg_worker.py

import asyncio
import httpx
from typing import Dict, List

class KGQueueWorker:
    """
    Background worker that processes KG queue
    """

    def __init__(self, db, kg_service_url: str):
        self.db = db
        self.kg_service_url = kg_service_url
        self.client = httpx.AsyncClient(timeout=300.0)
        self.running = False

    async def start(self):
        """Start processing queue"""
        self.running = True

        while self.running:
            # Get next pending item
            item = self.get_next_queued_item()

            if item:
                await self.process_item(item)
            else:
                # No items, wait before checking again
                await asyncio.sleep(5)

    def get_next_queued_item(self) -> Dict:
        """Get next item from queue (priority + FIFO)"""
        cursor = self.db.execute("""
            SELECT
                kq.id as queue_id,
                kq.content_id,
                cc.url,
                cc.title,
                cc.markdown,
                cc.tags
            FROM kg_processing_queue kq
            JOIN crawled_content cc ON kq.content_id = cc.id
            WHERE kq.status = 'pending'
              AND kq.retry_count < 3
            ORDER BY kq.priority DESC, kq.queued_at ASC
            LIMIT 1
        """)

        row = cursor.fetchone()
        if not row:
            return None

        queue_id, content_id, url, title, markdown, tags = row

        # Get chunk metadata
        chunks = self.get_chunk_metadata(content_id)

        return {
            "queue_id": queue_id,
            "content_id": content_id,
            "url": url,
            "title": title,
            "markdown": markdown,
            "tags": tags,
            "chunks": chunks
        }

    def get_chunk_metadata(self, content_id: int) -> List[Dict]:
        """Retrieve chunk positions from content_chunks table"""
        cursor = self.db.execute("""
            SELECT
                rowid,
                chunk_index,
                char_start,
                char_end,
                chunk_text
            FROM content_chunks
            WHERE content_id = ?
            ORDER BY chunk_index ASC
        """, (content_id,))

        chunks = []
        for row in cursor.fetchall():
            chunks.append({
                "vector_rowid": row[0],
                "chunk_index": row[1],
                "char_start": row[2],
                "char_end": row[3],
                "text": row[4]
            })

        return chunks

    async def process_item(self, item: Dict):
        """Send item to kg-service for processing"""
        queue_id = item["queue_id"]

        try:
            # Mark as processing
            self.db.execute(
                "UPDATE kg_processing_queue SET status='processing' WHERE id=?",
                (queue_id,)
            )
            self.db.commit()

            # POST to kg-service
            response = await self.client.post(
                f"{self.kg_service_url}/api/v1/ingest",
                json={
                    "content_id": item["content_id"],
                    "url": item["url"],
                    "title": item["title"],
                    "markdown": item["markdown"],
                    "chunks": item["chunks"],
                    "metadata": {
                        "tags": item["tags"],
                        "timestamp": datetime.now().isoformat()
                    }
                },
                timeout=300.0  # 5 minutes
            )

            response.raise_for_status()
            result = response.json()

            # Write results back to SQLite
            await self.write_back_results(item["content_id"], result)

            # Mark as completed
            self.db.execute("""
                UPDATE kg_processing_queue
                SET status='completed',
                    processed_at=CURRENT_TIMESTAMP,
                    result_summary=?
                WHERE id=?
            """, (json.dumps(result.get("summary")), queue_id))
            self.db.commit()

            logger.info(f"✓ Processed content_id={item['content_id']}: "
                       f"{result['entities_extracted']} entities, "
                       f"{result['relationships_extracted']} relationships")

        except Exception as e:
            logger.error(f"✗ Failed processing queue_id={queue_id}: {e}")

            # Increment retry count
            self.db.execute("""
                UPDATE kg_processing_queue
                SET status='failed',
                    retry_count=retry_count+1,
                    error_message=?
                WHERE id=?
            """, (str(e), queue_id))
            self.db.commit()

    async def write_back_results(self, content_id: int, result: Dict):
        """Write kg-service results back to SQLite"""
        # Update crawled_content
        self.db.execute("""
            UPDATE crawled_content
            SET kg_processed = 1,
                kg_entity_count = ?,
                kg_relationship_count = ?,
                kg_document_id = ?,
                kg_processed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (
            result["entities_extracted"],
            result["relationships_extracted"],
            result["neo4j_document_id"],
            content_id
        ))

        # Insert chunk_entities
        for entity in result.get("entities", []):
            for appearance in entity.get("chunk_appearances", []):
                self.db.execute("""
                    INSERT INTO chunk_entities (
                        chunk_rowid,
                        content_id,
                        entity_text,
                        entity_type_primary,
                        entity_type_sub1,
                        entity_type_sub2,
                        entity_type_sub3,
                        confidence,
                        offset_start,
                        offset_end,
                        neo4j_node_id,
                        spans_multiple_chunks
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    appearance["vector_rowid"],
                    content_id,
                    entity["text"],
                    entity["type_primary"],
                    entity.get("type_sub1"),
                    entity.get("type_sub2"),
                    entity.get("type_sub3"),
                    entity["confidence"],
                    appearance["offset_start"],
                    appearance["offset_end"],
                    entity["neo4j_node_id"],
                    len(entity.get("chunk_appearances", [])) > 1
                ))

        # Insert chunk_relationships
        for rel in result.get("relationships", []):
            chunk_ids_json = json.dumps(rel.get("chunk_rowids", []))

            self.db.execute("""
                INSERT INTO chunk_relationships (
                    content_id,
                    subject_entity,
                    predicate,
                    object_entity,
                    confidence,
                    context,
                    neo4j_relationship_id,
                    spans_chunks,
                    chunk_rowids
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                content_id,
                rel["subject_text"],
                rel["predicate"],
                rel["object_text"],
                rel["confidence"],
                rel["context"],
                rel.get("neo4j_relationship_id"),
                rel["spans_chunks"],
                chunk_ids_json
            ))

        self.db.commit()
        logger.info(f"✓ Wrote back {len(result['entities'])} entities, "
                   f"{len(result['relationships'])} relationships")
```

**Process:**
1. Poll queue table every 5 seconds
2. Get next pending item (priority + FIFO)
3. Retrieve full markdown + chunk metadata from SQLite
4. POST to kg-service API
5. Wait for response (up to 5 minutes timeout)
6. Write results back to SQLite
7. Mark queue item as completed/failed

**Error Handling:**
- Retry up to 3 times on failure
- Exponential backoff between retries
- Log errors to `kg_processing_errors` table
- Alert on persistent failures

**Startup:**
```python
# In mcpragcrawl4ai startup (core/rag_processor.py or api/api.py)

from core.workers.kg_worker import KGQueueWorker

# Start background worker
if settings.KG_SERVICE_ENABLED:
    worker = KGQueueWorker(
        db=GLOBAL_DB.db,
        kg_service_url=settings.KG_SERVICE_URL
    )
    asyncio.create_task(worker.start())
    logger.info("✓ KG queue worker started")
```

**Code Location:**
- NEW: `mcpragcrawl4ai/core/workers/kg_worker.py`
- NEW: `mcpragcrawl4ai/core/data/kg_queue.py` (queue utilities)

---

#### Step 5: kg-service Receives Request

**API Endpoint:** `POST http://kg-service:8088/api/v1/ingest`

**Request Format:**
```json
{
  "content_id": 123,
  "url": "https://docs.fastapi.com",
  "title": "FastAPI Documentation",
  "markdown": "# FastAPI\n\nFastAPI is a modern web framework...\n\n[10,000+ words of full document]",
  "chunks": [
    {
      "vector_rowid": 45001,
      "chunk_index": 0,
      "char_start": 0,
      "char_end": 2500,
      "text": "# FastAPI\n\nFastAPI is a modern..."
    },
    {
      "vector_rowid": 45002,
      "chunk_index": 1,
      "char_start": 2450,
      "char_end": 4950,
      "text": "...modern web framework for Python..."
    }
    // ... all chunks with actual boundaries
  ],
  "metadata": {
    "tags": "python,api,web",
    "timestamp": "2025-10-15T12:00:00Z",
    "source": "mcpragcrawl4ai"
  }
}
```

**Request Validation:**
```python
# kg-service/api/models.py

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional

class ChunkMetadata(BaseModel):
    vector_rowid: int = Field(..., description="SQLite content_vectors rowid")
    chunk_index: int = Field(..., ge=0, description="Sequential chunk number")
    char_start: int = Field(..., ge=0, description="Character position in original markdown")
    char_end: int = Field(..., gt=0, description="Character end position")
    text: str = Field(..., min_length=10, description="Actual chunk text")

    @validator('char_end')
    def end_after_start(cls, v, values):
        if 'char_start' in values and v <= values['char_start']:
            raise ValueError('char_end must be greater than char_start')
        return v

class IngestRequest(BaseModel):
    content_id: int = Field(..., gt=0, description="mcpragcrawl4ai content ID")
    url: str = Field(..., max_length=2048, description="Source URL")
    title: str = Field(..., max_length=500, description="Document title")
    markdown: str = Field(..., min_length=50, max_length=1000000, description="Full document markdown")
    chunks: List[ChunkMetadata] = Field(..., min_items=1, description="Chunk boundaries")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional metadata")

    @validator('chunks')
    def validate_chunks(cls, v):
        # Ensure chunks are ordered
        for i in range(len(v) - 1):
            if v[i].chunk_index >= v[i+1].chunk_index:
                raise ValueError('Chunks must be ordered by chunk_index')
        return v

class IngestResponse(BaseModel):
    success: bool
    content_id: int
    neo4j_document_id: str
    entities_extracted: int
    relationships_extracted: int
    processing_time_ms: int
    entities: List[Dict]  # Detailed entity data
    relationships: List[Dict]  # Detailed relationship data
    summary: Dict[str, int]  # Statistics
```

**Code Location:**
- `kg-service/api/models.py` (Pydantic models)
- `kg-service/api/routes.py` (endpoint implementation)

---

## 4. Database Schema Design

### 4.1 mcpragcrawl4ai Schema Changes

#### New Table: content_chunks

**Purpose:** Track chunk metadata (positions, boundaries)

```sql
CREATE TABLE content_chunks (
    rowid INTEGER PRIMARY KEY,          -- Same as content_vectors rowid
    content_id INTEGER NOT NULL,        -- FK to crawled_content
    chunk_index INTEGER NOT NULL,       -- Sequential: 0, 1, 2, ...
    chunk_text TEXT NOT NULL,           -- Full chunk text
    char_start INTEGER NOT NULL,        -- Position in original markdown
    char_end INTEGER NOT NULL,          -- End position in original markdown
    word_count INTEGER,                 -- Words in chunk
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (content_id) REFERENCES crawled_content(id) ON DELETE CASCADE
);

CREATE INDEX idx_content_chunks_content_id ON content_chunks(content_id);
CREATE INDEX idx_content_chunks_position ON content_chunks(char_start, char_end);
```

**Example Data:**
```
rowid | content_id | chunk_index | char_start | char_end | word_count
45001 |        123 |           0 |          0 |     2500 |        500
45002 |        123 |           1 |       2450 |     4950 |        500
45003 |        123 |           2 |       4900 |     7400 |        500
```

---

#### New Table: chunk_entities

**Purpose:** Map entities to specific chunks

```sql
CREATE TABLE chunk_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_rowid INTEGER NOT NULL,       -- Which chunk (FK to content_chunks)
    content_id INTEGER NOT NULL,        -- Which document (FK to crawled_content)
    entity_text TEXT NOT NULL,          -- "FastAPI", "Python", etc.
    entity_normalized TEXT,             -- lowercase for deduplication
    entity_type_primary TEXT,           -- "Framework", "Language", etc.
    entity_type_sub1 TEXT,              -- "Backend", "Interpreted", etc.
    entity_type_sub2 TEXT,              -- "Python", "DynamicTyped", etc.
    entity_type_sub3 TEXT,              -- Optional 3rd level
    confidence REAL,                    -- 0.0-1.0 from GLiNER
    offset_start INTEGER,               -- Position within chunk
    offset_end INTEGER,                 -- End position within chunk
    neo4j_node_id TEXT,                 -- Reference to Neo4j Entity node
    spans_multiple_chunks BOOLEAN DEFAULT 0,  -- True if entity in overlap region
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (chunk_rowid) REFERENCES content_chunks(rowid) ON DELETE CASCADE,
    FOREIGN KEY (content_id) REFERENCES crawled_content(id) ON DELETE CASCADE
);

CREATE INDEX idx_chunk_entities_chunk ON chunk_entities(chunk_rowid);
CREATE INDEX idx_chunk_entities_entity ON chunk_entities(entity_text);
CREATE INDEX idx_chunk_entities_type ON chunk_entities(entity_type_primary, entity_type_sub1);
CREATE INDEX idx_chunk_entities_content ON chunk_entities(content_id);
CREATE INDEX idx_chunk_entities_neo4j ON chunk_entities(neo4j_node_id);
```

**Example Data:**
```
id | chunk_rowid | content_id | entity_text | type_primary | type_sub1 | type_sub2 | confidence | offset_start | offset_end | neo4j_node_id
1  |       45001 |        123 | FastAPI     | Framework    | Backend   | Python    |       0.95 |          342 |        349 | entity_f4s7t9
2  |       45001 |        123 | Pydantic    | Framework    | Data      | Python    |       0.92 |          523 |        531 | entity_p8y2d1
3  |       45002 |        123 | FastAPI     | Framework    | Backend   | Python    |       0.95 |           73 |         80 | entity_f4s7t9
```

Note: "FastAPI" appears in rows 1 and 3 (different chunks, same entity)

---

#### New Table: chunk_relationships

**Purpose:** Track relationships and which chunks they appear in

```sql
CREATE TABLE chunk_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_id INTEGER NOT NULL,        -- Which document
    subject_entity TEXT NOT NULL,       -- "FastAPI"
    predicate TEXT NOT NULL,            -- "uses", "competes_with", etc.
    object_entity TEXT NOT NULL,        -- "Pydantic"
    confidence REAL,                    -- 0.0-1.0 from vLLM
    context TEXT,                       -- Sentence where relationship found
    neo4j_relationship_id TEXT,         -- Reference to Neo4j relationship
    spans_chunks BOOLEAN DEFAULT 0,     -- True if entities in different chunks
    chunk_rowids TEXT,                  -- JSON array: [45001] or [45001, 45015]
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (content_id) REFERENCES crawled_content(id) ON DELETE CASCADE
);

CREATE INDEX idx_chunk_relationships_content ON chunk_relationships(content_id);
CREATE INDEX idx_chunk_relationships_subject ON chunk_relationships(subject_entity);
CREATE INDEX idx_chunk_relationships_object ON chunk_relationships(object_entity);
CREATE INDEX idx_chunk_relationships_predicate ON chunk_relationships(predicate);
```

**Example Data:**
```
id | content_id | subject_entity | predicate      | object_entity | confidence | context                          | spans_chunks | chunk_rowids
1  |        123 | FastAPI        | uses           | Pydantic      |       0.88 | FastAPI uses Pydantic for...    |            0 | [45001]
2  |        123 | FastAPI        | competes_with  | Django        |       0.72 | FastAPI is faster than Django... |            1 | [45001,45015]
```

---

#### New Table: kg_processing_queue

**Purpose:** Queue documents for KG processing

```sql
CREATE TABLE kg_processing_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_id INTEGER NOT NULL,
    status TEXT DEFAULT 'pending',      -- pending, processing, completed, failed
    priority INTEGER DEFAULT 1,         -- Higher = process first
    queued_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    processing_started_at DATETIME,
    processed_at DATETIME,
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    result_summary TEXT,                -- JSON with statistics

    FOREIGN KEY (content_id) REFERENCES crawled_content(id) ON DELETE CASCADE
);

CREATE INDEX idx_kg_queue_status ON kg_processing_queue(status, priority, queued_at);
```

---

#### Modified Table: crawled_content

**Add KG tracking columns:**

```sql
ALTER TABLE crawled_content ADD COLUMN kg_processed BOOLEAN DEFAULT 0;
ALTER TABLE crawled_content ADD COLUMN kg_entity_count INTEGER DEFAULT 0;
ALTER TABLE crawled_content ADD COLUMN kg_relationship_count INTEGER DEFAULT 0;
ALTER TABLE crawled_content ADD COLUMN kg_document_id TEXT;       -- Neo4j Document node ID
ALTER TABLE crawled_content ADD COLUMN kg_processed_at DATETIME;

CREATE INDEX idx_crawled_content_kg ON crawled_content(kg_processed, kg_processed_at);
```

**Migration Script:**

```python
# mcpragcrawl4ai/migrations/001_add_kg_support.py

def upgrade(db):
    """Add KG support to existing database"""

    # Add columns to crawled_content
    db.execute("ALTER TABLE crawled_content ADD COLUMN kg_processed BOOLEAN DEFAULT 0")
    db.execute("ALTER TABLE crawled_content ADD COLUMN kg_entity_count INTEGER DEFAULT 0")
    db.execute("ALTER TABLE crawled_content ADD COLUMN kg_relationship_count INTEGER DEFAULT 0")
    db.execute("ALTER TABLE crawled_content ADD COLUMN kg_document_id TEXT")
    db.execute("ALTER TABLE crawled_content ADD COLUMN kg_processed_at DATETIME")
    db.execute("CREATE INDEX idx_crawled_content_kg ON crawled_content(kg_processed, kg_processed_at)")

    # Create new tables
    db.executescript('''
        CREATE TABLE content_chunks ( ... );
        CREATE TABLE chunk_entities ( ... );
        CREATE TABLE chunk_relationships ( ... );
        CREATE TABLE kg_processing_queue ( ... );
    ''')

    db.commit()
    print("✓ Database upgraded for KG support")

def downgrade(db):
    """Remove KG support (for rollback)"""
    # Implementation for reverting changes
    pass
```

---

### 4.2 Neo4j Graph Schema

#### Node Types

**1. Document Node**

```cypher
(:Document {
    content_id: INTEGER,        -- mcpragcrawl4ai ID (unique)
    url: STRING,
    title: STRING,
    markdown_hash: STRING,      -- SHA256 for change detection
    word_count: INTEGER,
    chunk_count: INTEGER,
    tags: [STRING],
    processed_at: DATETIME,
    source: STRING              -- "mcpragcrawl4ai"
})

CREATE CONSTRAINT document_content_id IF NOT EXISTS
FOR (d:Document) REQUIRE d.content_id IS UNIQUE;

CREATE INDEX document_url IF NOT EXISTS
FOR (d:Document) ON (d.url);
```

---

**2. Chunk Node**

```cypher
(:Chunk {
    vector_rowid: INTEGER,      -- mcpragcrawl4ai content_vectors rowid (unique)
    content_id: INTEGER,        -- parent document
    chunk_index: INTEGER,       -- sequential position
    char_start: INTEGER,
    char_end: INTEGER,
    word_count: INTEGER,
    text_snippet: STRING        -- first 200 chars for preview
})

CREATE CONSTRAINT chunk_rowid IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.vector_rowid IS UNIQUE;

CREATE INDEX chunk_content_id IF NOT EXISTS
FOR (c:Chunk) ON (c.content_id);
```

---

**3. Entity Node**

```cypher
(:Entity {
    text: STRING,               -- "FastAPI", "Python"
    normalized: STRING,         -- lowercase for matching
    type_primary: STRING,       -- "Framework", "Language"
    type_sub1: STRING,          -- "Backend", "Interpreted"
    type_sub2: STRING,          -- "Python", "DynamicTyped"
    type_sub3: STRING,          -- Optional 3rd level
    total_mentions: INTEGER,    -- Across all documents
    document_count: INTEGER,    -- Number of documents mentioning
    first_seen: DATETIME,
    last_seen: DATETIME
})

CREATE CONSTRAINT entity_text_type IF NOT EXISTS
FOR (e:Entity) REQUIRE (e.text, e.type_primary, e.type_sub1, e.type_sub2) IS UNIQUE;

CREATE INDEX entity_text IF NOT EXISTS
FOR (e:Entity) ON (e.text);

CREATE INDEX entity_type_hierarchy IF NOT EXISTS
FOR (e:Entity) ON (e.type_primary, e.type_sub1, e.type_sub2);

CREATE INDEX entity_normalized IF NOT EXISTS
FOR (e:Entity) ON (e.normalized);
```

---

#### Relationship Types

**1. HAS_CHUNK (Document → Chunk)**

```cypher
(:Document)-[:HAS_CHUNK {
    index: INTEGER,             -- Sequential chunk number
    created_at: DATETIME
}]->(:Chunk)
```

---

**2. CONTAINS (Chunk → Entity)**

```cypher
(:Chunk)-[:CONTAINS {
    offset_start: INTEGER,      -- Position within chunk
    offset_end: INTEGER,
    confidence: FLOAT,          -- From GLiNER
    context_before: STRING,     -- 50 chars before
    context_after: STRING,      -- 50 chars after
    sentence: STRING            -- Full sentence
}]->(:Entity)
```

---

**3. RELATED_TO (Entity → Entity)**

```cypher
(:Entity)-[:RELATED_TO {
    predicate: STRING,          -- "uses", "competes_with", "implements", etc.
    confidence: FLOAT,          -- From vLLM
    context: STRING,            -- Sentence where found
    document_id: INTEGER,       -- Which document (for filtering)
    chunk_ids: [INTEGER],       -- Which chunks [45001] or [45001, 45015]
    spans_chunks: BOOLEAN,      -- True if crosses chunk boundary
    extraction_method: STRING,  -- "vllm", "pattern", "dependency"
    created_at: DATETIME
}]->(:Entity)

CREATE INDEX relationship_predicate IF NOT EXISTS
FOR ()-[r:RELATED_TO]-() ON (r.predicate);

CREATE INDEX relationship_document IF NOT EXISTS
FOR ()-[r:RELATED_TO]-() ON (r.document_id);
```

---

**4. CO_OCCURS_WITH (Entity → Entity)**

**Purpose:** Track entities that frequently appear together (calculated during ingestion)

```cypher
(:Entity)-[:CO_OCCURS_WITH {
    count: INTEGER,             -- Number of co-occurrences
    document_count: INTEGER,    -- Number of documents
    avg_proximity: FLOAT,       -- Average chars between mentions
    documents: [INTEGER]        -- content_ids where they co-occur
}]->(:Entity)
```

---

#### Graph Schema Initialization

```cypher
// kg-service/storage/schema.cypher

// Create constraints (run once)
CREATE CONSTRAINT document_content_id IF NOT EXISTS
FOR (d:Document) REQUIRE d.content_id IS UNIQUE;

CREATE CONSTRAINT chunk_rowid IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.vector_rowid IS UNIQUE;

CREATE CONSTRAINT entity_text_type IF NOT EXISTS
FOR (e:Entity) REQUIRE (e.text, e.type_primary, e.type_sub1, e.type_sub2) IS UNIQUE;

// Create indexes for performance
CREATE INDEX document_url IF NOT EXISTS FOR (d:Document) ON (d.url);
CREATE INDEX chunk_content_id IF NOT EXISTS FOR (c:Chunk) ON (c.content_id);
CREATE INDEX entity_text IF NOT EXISTS FOR (e:Entity) ON (e.text);
CREATE INDEX entity_type_hierarchy IF NOT EXISTS FOR (e:Entity) ON (e.type_primary, e.type_sub1, e.type_sub2);
CREATE INDEX entity_normalized IF NOT EXISTS FOR (e:Entity) ON (e.normalized);
CREATE INDEX relationship_predicate IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.predicate);
CREATE INDEX relationship_document IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.document_id);

// Full-text search indexes
CALL db.index.fulltext.createNodeIndex(
    "entityTextSearch",
    ["Entity"],
    ["text", "type_primary", "type_sub1"]
) IF NOT EXISTS;
```

**Code Location:** `kg-service/storage/schema.py` (Python wrapper for schema setup)

---

## 5. Entity Extraction System

### 5.1 GLiNER Configuration

**Model:** `urchade/gliner_large-v2.1`

**Capabilities:**
- Zero-shot entity extraction
- Can handle 300+ entity types simultaneously
- Hierarchical type classification
- Confidence scoring per entity

**Settings:**
```python
# kg-service/config.py

GLINER_MODEL = "urchade/gliner_large-v2.1"
GLINER_THRESHOLD = 0.5          # Minimum confidence
GLINER_BATCH_SIZE = 8           # Process N texts at once
GLINER_MAX_LENGTH = 384         # Token limit per batch
```

### 5.2 Entity Extraction Process

**Input:** Full markdown document (10,000+ words)

**Process:**

```python
# kg-service/extractors/entity_extractor.py

from gliner import GLiNER
import yaml

class EntityExtractor:
    def __init__(self):
        self.model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
        self.entity_types = self._load_taxonomy()

    def _load_taxonomy(self):
        """Load 300+ entity types from YAML"""
        with open("taxonomy/entities.yaml") as f:
            taxonomy = yaml.safe_load(f)

        types = []
        for category in taxonomy["entity_categories"].values():
            types.extend(category)

        return types  # ["Framework::Backend::Python", "Language::Interpreted::DynamicTyped", ...]

    def extract(self, text: str) -> List[Dict]:
        """
        Extract entities from full document

        Returns:
            [
                {
                    "text": "FastAPI",
                    "start": 342,
                    "end": 349,
                    "type_primary": "Framework",
                    "type_sub1": "Backend",
                    "type_sub2": "Python",
                    "type_sub3": None,
                    "confidence": 0.95,
                    "context_before": "modern web ",
                    "context_after": " for building",
                    "sentence": "FastAPI is a modern web framework for building APIs."
                },
                ...
            ]
        """

        # GLiNER prediction
        predictions = self.model.predict_entities(
            text,
            self.entity_types,
            threshold=0.5
        )

        entities = []
        for pred in predictions:
            # Parse hierarchical type
            type_parts = pred["label"].split("::")

            # Get context
            context = self._extract_context(text, pred["start"], pred["end"])

            entity = {
                "text": pred["text"],
                "normalized": pred["text"].lower().strip(),
                "start": pred["start"],
                "end": pred["end"],

                # Type hierarchy
                "type_full": pred["label"],
                "type_primary": type_parts[0] if len(type_parts) > 0 else None,
                "type_sub1": type_parts[1] if len(type_parts) > 1 else None,
                "type_sub2": type_parts[2] if len(type_parts) > 2 else None,
                "type_sub3": type_parts[3] if len(type_parts) > 3 else None,

                # Metadata
                "confidence": float(pred["score"]),
                **context
            }

            entities.append(entity)

        return entities

    def _extract_context(self, text: str, start: int, end: int, window: int = 50):
        """Extract surrounding context for entity mention"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)

        # Get sentence boundaries
        sentence_start = text.rfind('.', 0, start) + 1
        sentence_end = text.find('.', end)
        if sentence_end == -1:
            sentence_end = len(text)

        return {
            "context_before": text[context_start:start].strip(),
            "context_after": text[end:context_end].strip(),
            "sentence": text[sentence_start:sentence_end].strip()
        }
```

**Output:**
```python
[
    {
        "text": "FastAPI",
        "normalized": "fastapi",
        "start": 342,
        "end": 349,
        "type_primary": "Framework",
        "type_sub1": "Backend",
        "type_sub2": "Python",
        "type_sub3": None,
        "confidence": 0.95,
        "context_before": "modern web ",
        "context_after": " for building",
        "sentence": "FastAPI is a modern web framework for building APIs."
    },
    # ... 100+ more entities
]
```

---

## 6. Relationship Extraction System

### 6.1 vLLM Configuration

**Auto-Discovery:**
- Query `http://localhost:8078/v1/models` on startup
- Store model name (e.g., "Qwen/Qwen2.5-7B-Instruct")
- Reset to `None` on connection failure
- Retry every 30 seconds

**Settings:**
```python
# kg-service/config.py

VLLM_BASE_URL = "http://localhost:8078"
VLLM_TIMEOUT = 120                  # seconds
VLLM_MAX_TOKENS = 4096
VLLM_TEMPERATURE = 0.1              # Low for consistent extraction
VLLM_RETRY_INTERVAL = 30            # seconds
```

### 6.2 Relationship Extraction Process

**Input:**
- Full markdown document
- List of extracted entities with positions

**Process:**

```python
# kg-service/extractors/relation_extractor.py

from clients.vllm_client import VLLMClient
import json

class RelationshipExtractor:
    def __init__(self):
        self.vllm_client = VLLMClient()
        self.relationship_types = [
            "uses", "implements", "extends", "depends_on",
            "competes_with", "replaces", "precedes",
            "part_of", "has_component", "requires",
            "compatible_with", "built_with", "runs_on"
        ]

    async def extract(self, text: str, entities: List[Dict]) -> List[Dict]:
        """
        Extract relationships using LLM reasoning

        Args:
            text: Full markdown document
            entities: List of entities with positions

        Returns:
            [
                {
                    "subject": "FastAPI",
                    "predicate": "uses",
                    "object": "Pydantic",
                    "confidence": 0.88,
                    "context": "FastAPI uses Pydantic for data validation",
                    "evidence_start": 523,
                    "evidence_end": 570
                },
                ...
            ]
        """

        # Build entity list for prompt
        entity_mentions = "\n".join([
            f"- {e['text']} ({e['type_primary']}::{e['type_sub1']}::{e['type_sub2']}) at position {e['start']}"
            for e in entities[:50]  # Limit to top 50 entities to avoid token limits
        ])

        # Build prompt
        prompt = self._build_relationship_prompt(text, entity_mentions)

        # Query vLLM
        try:
            response = await self.vllm_client.extract_json(
                prompt=prompt,
                max_tokens=4096,
                temperature=0.1
            )

            relationships = response.get("relationships", [])

            # Validate and enrich
            validated = []
            for rel in relationships:
                if self._validate_relationship(rel, entities):
                    enriched = self._enrich_relationship(rel, text, entities)
                    validated.append(enriched)

            return validated

        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            # Fall back to pattern-based extraction
            return self._fallback_extraction(text, entities)

    def _build_relationship_prompt(self, text: str, entity_mentions: str) -> str:
        """Build LLM prompt for relationship extraction"""

        return f"""You are an expert at extracting semantic relationships between entities in technical documentation.

Given the following document and entities, identify relationships between them.

**Entities found:**
{entity_mentions}

**Document excerpt:**
{text[:4000]}  # First 4000 chars for context

**Task:**
For each relationship you find, output JSON in this format:

{{
  "relationships": [
    {{
      "subject": "EntityName",
      "predicate": "relationship_type",
      "object": "EntityName",
      "confidence": 0.85,
      "context": "exact sentence or phrase where this relationship is stated"
    }}
  ]
}}

**Relationship types to look for:**
- uses: X uses Y (e.g., "FastAPI uses Pydantic")
- implements: X implements Y (e.g., "FastAPI implements ASGI")
- depends_on: X depends on Y
- competes_with: X competes with Y (e.g., "FastAPI vs Django")
- built_with: X built with Y (e.g., "API built with FastAPI")
- runs_on: X runs on Y (e.g., "FastAPI runs on Uvicorn")
- compatible_with: X compatible with Y

**Important:**
- Only extract relationships between entities in the entity list above
- Include exact context (sentence) where relationship was found
- Confidence: 0.9+ for explicit statements, 0.7-0.9 for implied, 0.5-0.7 for weak
- Focus on technical relationships, not generic descriptors

Output valid JSON only (no markdown formatting):"""

    def _validate_relationship(self, rel: Dict, entities: List[Dict]) -> bool:
        """Validate relationship has valid subject/object entities"""
        entity_texts = {e["text"] for e in entities}

        return (
            rel.get("subject") in entity_texts and
            rel.get("object") in entity_texts and
            rel.get("predicate") in self.relationship_types and
            rel.get("confidence", 0) >= 0.5
        )

    def _enrich_relationship(self, rel: Dict, text: str, entities: List[Dict]) -> Dict:
        """Add entity positions and metadata to relationship"""

        # Find entity objects
        subject_entity = next((e for e in entities if e["text"] == rel["subject"]), None)
        object_entity = next((e for e in entities if e["text"] == rel["object"]), None)

        if not subject_entity or not object_entity:
            return rel

        # Find context position in text
        context = rel.get("context", "")
        context_start = text.find(context) if context else -1
        context_end = context_start + len(context) if context_start != -1 else -1

        return {
            **rel,
            "subject_type_primary": subject_entity["type_primary"],
            "subject_type_sub1": subject_entity["type_sub1"],
            "subject_position": subject_entity["start"],
            "object_type_primary": object_entity["type_primary"],
            "object_type_sub1": object_entity["type_sub1"],
            "object_position": object_entity["start"],
            "evidence_start": context_start,
            "evidence_end": context_end
        }

    def _fallback_extraction(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Pattern-based fallback if vLLM fails"""
        # Simple pattern matching for common relationships
        # e.g., "X uses Y", "X built with Y", etc.
        # Implementation: regex patterns or dependency parsing
        return []
```

**Output:**
```python
[
    {
        "subject": "FastAPI",
        "predicate": "uses",
        "object": "Pydantic",
        "confidence": 0.88,
        "context": "FastAPI uses Pydantic for data validation and serialization",
        "subject_type_primary": "Framework",
        "subject_position": 342,
        "object_type_primary": "Framework",
        "object_position": 523,
        "evidence_start": 510,
        "evidence_end": 570
    },
    # ... 50+ more relationships
]
```

---

## 7. Chunk Mapping System

### 7.1 Purpose

**Problem:** Entities/relationships extracted from full document (character positions 0-50,000)
**Solution:** Map them to specific chunks using actual chunk boundaries from mcpragcrawl4ai

### 7.2 Mapping Algorithm

```python
# kg-service/pipeline/chunk_mapper.py

class ChunkMapper:
    """Map entities and relationships to document chunks"""

    def map_entities_to_chunks(
        self,
        entities: List[Dict],
        chunks: List[Dict]
    ) -> List[Dict]:
        """
        Map entity positions to chunks

        Args:
            entities: [{text, start, end, ...}, ...]
            chunks: [{vector_rowid, char_start, char_end, ...}, ...]

        Returns:
            entities with added 'chunk_appearances' field
        """

        for entity in entities:
            entity_start = entity["start"]
            entity_end = entity["end"]

            # Find all chunks this entity appears in
            appearances = []

            for chunk in chunks:
                # Check if entity overlaps with chunk boundaries
                if self._overlaps(
                    entity_start, entity_end,
                    chunk["char_start"], chunk["char_end"]
                ):
                    # Calculate offset within chunk
                    offset_start = max(0, entity_start - chunk["char_start"])
                    offset_end = min(
                        chunk["char_end"] - chunk["char_start"],
                        entity_end - chunk["char_start"]
                    )

                    appearances.append({
                        "vector_rowid": chunk["vector_rowid"],
                        "chunk_index": chunk["chunk_index"],
                        "offset_start": offset_start,
                        "offset_end": offset_end
                    })

            entity["chunk_appearances"] = appearances
            entity["spans_multiple_chunks"] = len(appearances) > 1

        return entities

    def map_relationships_to_chunks(
        self,
        relationships: List[Dict],
        entities: List[Dict]
    ) -> List[Dict]:
        """
        Map relationships to chunks based on entity positions

        Args:
            relationships: [{subject, predicate, object, ...}, ...]
            entities: Entities with chunk_appearances already mapped

        Returns:
            relationships with 'chunk_rowids' and 'spans_chunks' fields
        """

        # Build entity lookup
        entity_map = {e["text"]: e for e in entities}

        for rel in relationships:
            subject = entity_map.get(rel["subject"])
            object_ent = entity_map.get(rel["object"])

            if not subject or not object_ent:
                continue

            # Get chunks for subject and object
            subject_chunks = {a["vector_rowid"] for a in subject.get("chunk_appearances", [])}
            object_chunks = {a["vector_rowid"] for a in object_ent.get("chunk_appearances", [])}

            # Relationship is in chunks where BOTH entities appear
            common_chunks = subject_chunks & object_chunks

            if common_chunks:
                # Entities in same chunk(s)
                rel["chunk_rowids"] = list(common_chunks)
                rel["spans_chunks"] = False
            else:
                # Entities in different chunks (relationship spans boundaries)
                rel["chunk_rowids"] = list(subject_chunks | object_chunks)
                rel["spans_chunks"] = True

        return relationships

    def _overlaps(self, start1: int, end1: int, start2: int, end2: int) -> bool:
        """Check if two ranges overlap"""
        return max(start1, start2) < min(end1, end2)
```

### 7.3 Example Mapping

**Input:**
```python
entity = {
    "text": "FastAPI",
    "start": 2480,  # In original document
    "end": 2487
}

chunks = [
    {"vector_rowid": 45001, "char_start": 0, "char_end": 2500},
    {"vector_rowid": 45002, "char_start": 2450, "char_end": 4950},  # Overlap region
    {"vector_rowid": 45003, "char_start": 4900, "char_end": 7400}
]
```

**Processing:**
- Entity position: 2480-2487
- Chunk 1: 0-2500 → **OVERLAPS** (entity at 2480)
- Chunk 2: 2450-4950 → **OVERLAPS** (entity appears at start due to overlap)
- Chunk 3: 4900-7400 → NO OVERLAP

**Output:**
```python
entity = {
    "text": "FastAPI",
    "start": 2480,
    "end": 2487,
    "chunk_appearances": [
        {"vector_rowid": 45001, "chunk_index": 0, "offset_start": 2480, "offset_end": 2487},
        {"vector_rowid": 45002, "chunk_index": 1, "offset_start": 30, "offset_end": 37}
    ],
    "spans_multiple_chunks": True
}
```

**Relationship Example:**
```python
relationship = {
    "subject": "FastAPI",     # In chunks [45001, 45002]
    "predicate": "competes_with",
    "object": "Django"        # In chunks [45015]
}

# After mapping:
relationship = {
    "subject": "FastAPI",
    "predicate": "competes_with",
    "object": "Django",
    "chunk_rowids": [45001, 45002, 45015],
    "spans_chunks": True      # Entities in different chunks
}
```

---

## 8. Neo4j Graph Storage

### 8.1 Storage Process

```python
# kg-service/storage/neo4j_client.py

from neo4j import GraphDatabase
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def store_document_and_graph(
        self,
        content_id: int,
        url: str,
        title: str,
        markdown_hash: str,
        chunks: List[Dict],
        entities: List[Dict],
        relationships: List[Dict],
        metadata: Dict
    ) -> Dict:
        """
        Store complete document graph in Neo4j

        Returns:
            {
                "document_id": "uuid",
                "entities_created": 50,
                "relationships_created": 30,
                "chunks_created": 20
            }
        """

        with self.driver.session() as session:
            result = session.write_transaction(
                self._create_document_graph,
                content_id, url, title, markdown_hash,
                chunks, entities, relationships, metadata
            )
            return result

    def _create_document_graph(
        self, tx,
        content_id, url, title, markdown_hash,
        chunks, entities, relationships, metadata
    ):
        """Transaction: Create complete document graph"""

        # Step 1: Create/update Document node
        doc_result = tx.run("""
            MERGE (d:Document {content_id: $content_id})
            ON CREATE SET
                d.url = $url,
                d.title = $title,
                d.markdown_hash = $markdown_hash,
                d.chunk_count = $chunk_count,
                d.tags = $tags,
                d.processed_at = datetime(),
                d.source = 'mcpragcrawl4ai'
            ON MATCH SET
                d.markdown_hash = $markdown_hash,
                d.chunk_count = $chunk_count,
                d.processed_at = datetime()
            RETURN elementId(d) as doc_id
        """, {
            "content_id": content_id,
            "url": url,
            "title": title,
            "markdown_hash": markdown_hash,
            "chunk_count": len(chunks),
            "tags": metadata.get("tags", "").split(",")
        })

        doc_id = doc_result.single()["doc_id"]

        # Step 2: Create Chunk nodes
        chunks_created = 0
        for chunk in chunks:
            tx.run("""
                MERGE (c:Chunk {vector_rowid: $vector_rowid})
                ON CREATE SET
                    c.content_id = $content_id,
                    c.chunk_index = $chunk_index,
                    c.char_start = $char_start,
                    c.char_end = $char_end,
                    c.word_count = $word_count,
                    c.text_snippet = $text_snippet

                WITH c
                MATCH (d:Document {content_id: $content_id})
                MERGE (d)-[:HAS_CHUNK {index: $chunk_index}]->(c)
            """, {
                "vector_rowid": chunk["vector_rowid"],
                "content_id": content_id,
                "chunk_index": chunk["chunk_index"],
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
                "word_count": chunk.get("word_count", 0),
                "text_snippet": chunk["text"][:200]  # First 200 chars
            })
            chunks_created += 1

        # Step 3: Create/update Entity nodes and CONTAINS relationships
        entities_created = 0
        entity_id_map = {}  # Map entity text to Neo4j ID

        for entity in entities:
            result = tx.run("""
                MERGE (e:Entity {
                    text: $text,
                    type_primary: $type_primary,
                    type_sub1: $type_sub1,
                    type_sub2: $type_sub2
                })
                ON CREATE SET
                    e.normalized = $normalized,
                    e.type_sub3 = $type_sub3,
                    e.total_mentions = 1,
                    e.document_count = 1,
                    e.first_seen = datetime(),
                    e.last_seen = datetime()
                ON MATCH SET
                    e.total_mentions = e.total_mentions + $appearance_count,
                    e.last_seen = datetime()

                RETURN elementId(e) as entity_id
            """, {
                "text": entity["text"],
                "normalized": entity["normalized"],
                "type_primary": entity["type_primary"],
                "type_sub1": entity.get("type_sub1"),
                "type_sub2": entity.get("type_sub2"),
                "type_sub3": entity.get("type_sub3"),
                "appearance_count": len(entity.get("chunk_appearances", []))
            })

            entity_id = result.single()["entity_id"]
            entity_id_map[entity["text"]] = entity_id
            entities_created += 1

            # Create CONTAINS relationships for each chunk appearance
            for appearance in entity.get("chunk_appearances", []):
                tx.run("""
                    MATCH (c:Chunk {vector_rowid: $vector_rowid})
                    MATCH (e:Entity) WHERE elementId(e) = $entity_id
                    MERGE (c)-[r:CONTAINS {offset_start: $offset_start}]->(e)
                    ON CREATE SET
                        r.offset_end = $offset_end,
                        r.confidence = $confidence,
                        r.context_before = $context_before,
                        r.context_after = $context_after,
                        r.sentence = $sentence
                """, {
                    "vector_rowid": appearance["vector_rowid"],
                    "entity_id": entity_id,
                    "offset_start": appearance["offset_start"],
                    "offset_end": appearance["offset_end"],
                    "confidence": entity["confidence"],
                    "context_before": entity.get("context_before", ""),
                    "context_after": entity.get("context_after", ""),
                    "sentence": entity.get("sentence", "")
                })

        # Step 4: Create RELATED_TO relationships
        relationships_created = 0

        for rel in relationships:
            subject_id = entity_id_map.get(rel["subject"])
            object_id = entity_id_map.get(rel["object"])

            if not subject_id or not object_id:
                continue

            result = tx.run("""
                MATCH (subj:Entity) WHERE elementId(subj) = $subject_id
                MATCH (obj:Entity) WHERE elementId(obj) = $object_id

                MERGE (subj)-[r:RELATED_TO {
                    predicate: $predicate,
                    document_id: $document_id
                }]->(obj)
                ON CREATE SET
                    r.confidence = $confidence,
                    r.context = $context,
                    r.chunk_ids = $chunk_ids,
                    r.spans_chunks = $spans_chunks,
                    r.extraction_method = 'vllm',
                    r.created_at = datetime()

                RETURN elementId(r) as rel_id
            """, {
                "subject_id": subject_id,
                "object_id": object_id,
                "predicate": rel["predicate"],
                "document_id": content_id,
                "confidence": rel.get("confidence", 0.5),
                "context": rel.get("context", ""),
                "chunk_ids": rel.get("chunk_rowids", []),
                "spans_chunks": rel.get("spans_chunks", False)
            })

            relationships_created += 1

        # Step 5: Calculate co-occurrences (entities appearing in same chunks)
        self._calculate_cooccurrences(tx, content_id)

        return {
            "document_id": doc_id,
            "entities_created": entities_created,
            "relationships_created": relationships_created,
            "chunks_created": chunks_created
        }

    def _calculate_cooccurrences(self, tx, content_id: int):
        """Calculate and store CO_OCCURS_WITH relationships"""

        tx.run("""
            // Find entities that appear in same chunks
            MATCH (c:Chunk {content_id: $content_id})-[:CONTAINS]->(e1:Entity)
            MATCH (c)-[:CONTAINS]->(e2:Entity)
            WHERE elementId(e1) < elementId(e2)  // Avoid duplicates

            WITH e1, e2, COUNT(DISTINCT c) as co_chunk_count
            WHERE co_chunk_count >= 2  // Must co-occur in at least 2 chunks

            MERGE (e1)-[co:CO_OCCURS_WITH]-(e2)
            ON CREATE SET
                co.count = co_chunk_count,
                co.document_count = 1,
                co.documents = [$content_id]
            ON MATCH SET
                co.count = co.count + co_chunk_count,
                co.document_count = co.document_count + 1,
                co.documents = co.documents + $content_id
        """, {"content_id": content_id})
```

---

## 9. SQLite Write-back System

### 9.1 Response Format from kg-service

```json
{
  "success": true,
  "content_id": 123,
  "neo4j_document_id": "4:abc123:456",
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
      "confidence": 0.95,
      "neo4j_node_id": "4:entity:789",
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
    // ... more entities
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
    // ... more relationships
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
    }
  }
}
```

### 9.2 Write-back Implementation

Already shown in [Step 4: Background Worker](#step-4-background-worker-mcpragcrawl4ai), specifically in the `write_back_results()` method.

**Summary:**
1. Update `crawled_content` table with KG metadata
2. Insert into `chunk_entities` for each entity appearance
3. Insert into `chunk_relationships` for each relationship
4. Commit transaction

---

## Summary: Part 1 Complete

This document covers:
- ✅ System architecture and components
- ✅ Complete data flow (crawl → chunk → queue → KG → Neo4j → write-back)
- ✅ Database schemas (SQLite + Neo4j)
- ✅ Entity extraction with GLiNER (300+ types)
- ✅ Relationship extraction with vLLM
- ✅ Chunk mapping algorithm
- ✅ Neo4j storage implementation
- ✅ SQLite write-back system

**Next Document: Part 2 will cover:**
- Query pipeline (Neo4j-first graph exploration)
- Graph traversal queries with configurable depth
- Vector search enhancement
- Result ranking and enrichment
- API design and Docker configuration
- Monitoring and performance tuning

---

**Status:** Ready for implementation of Part 1 components.
