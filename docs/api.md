---
layout: default
title: API Layer
---

# API Layer Documentation

## Module Overview

The API layer provides the external interface for document ingestion and knowledge retrieval. Built on FastAPI, it implements async request handling, Pydantic-based validation, structured error responses, and OpenAPI documentation.

**Files**:
- `api/server.py`: FastAPI application, lifecycle management, endpoints
- `api/models.py`: Pydantic request/response models
- `api/search_endpoints.py`: Entity search and chunk retrieval endpoints

## Module Architecture

### FastAPI Application (server.py)

**Purpose**: HTTP server orchestrating request routing, validation, processing delegation, and response formatting.

**Key Components**:

1. **Lifespan Manager**:
   - Handles startup and shutdown events
   - Initializes KGProcessor (connects to Neo4j, validates schema)
   - Checks vLLM availability (non-blocking, will retry on first request)
   - Closes connections gracefully on shutdown

2. **Middleware Stack**:
   - CORS middleware: Allows cross-origin requests (configurable origins)
   - Request logging middleware: Logs method, path, status, timing
   - Adds X-Process-Time header to all responses

3. **Exception Handlers**:
   - HTTPException handler: Returns structured error response
   - General exception handler: Catches unexpected errors, logs full traceback
   - All errors return ErrorResponse model with timestamp and error type

4. **Global State**:
   - `kg_processor`: Single KGProcessor instance for all requests
   - `processing_stats`: In-memory statistics (total processed, entity/relationship counts, timing)
   - `service_start_time`: Uptime tracking

**Initialization Sequence**:
```
1. Import configuration and validate settings
2. Create FastAPI app with lifespan context manager
3. Register search router for /api/v1/search/* endpoints
4. Add CORS middleware with wildcard origin (configure for production)
5. Add request logging middleware
6. On startup:
   - Validate critical settings (Neo4j URI, vLLM URL, taxonomy file)
   - Initialize KGProcessor (connects to Neo4j, runs schema init)
   - Check vLLM availability (logs warning if unavailable, continues)
7. Service ready for requests
```

### Request/Response Models (models.py)

**Purpose**: Type-safe contracts for API communication using Pydantic for validation.

**Model Hierarchy**:

#### Input Models

**ChunkMetadata**:
```
Represents a single chunk with vector database mapping.

Fields:
- vector_rowid: int (>0) - SQLite content_vectors rowid
- chunk_index: int (≥0) - Sequential chunk number
- char_start: int (≥0) - Character position in original markdown
- char_end: int (>char_start) - Character end position
- text: str (10-10000 chars) - Actual chunk text

Validators:
- end_after_start: Ensures char_end > char_start

Used by: IngestRequest
```

**IngestRequest**:
```
Main document ingestion payload from external systems.

Fields:
- content_id: int (>0) - Source system primary key
- url: str (max 2048) - Must start with http:// or https://
- title: str (max 500) - Document title
- markdown: str (50-1000000 chars) - Full document content
- chunks: List[ChunkMetadata] (1-1000) - Chunk boundaries
- metadata: Dict[str, Any] - Optional key-value pairs

Validators:
- validate_chunks_ordered: Ensures chunks sorted by chunk_index
- validate_url_format: Verifies URL starts with http/https

Used by: POST /api/v1/ingest
```

**EntitySearchRequest**:
```
Search entities by text matching.

Fields:
- entity_terms: List[str] (≥1) - Terms to search for
- limit: int (1-500, default 50) - Max results per term
- min_mentions: int (≥1, default 1) - Minimum mention count

Used by: POST /api/v1/search/entities
```

**ChunkSearchRequest**:
```
Find chunks containing specified entities.

Fields:
- entity_ids: Optional[List[str]] - Neo4j element IDs
- entity_names: Optional[List[str]] - Entity text names
- limit: int (1-1000, default 100) - Max chunks
- include_document_info: bool (default True) - Include URL/title

Validation: Must provide either entity_ids or entity_names

Used by: POST /api/v1/search/chunks
```

**EntityExpansionRequest**:
```
Discover related entities via graph traversal.

Fields:
- entity_names: List[str] (≥1) - Starting entities
- max_expansions: int (1-100, default 10) - Max related entities
- min_confidence: float (0-1, default 0.3) - Relationship confidence threshold
- expansion_depth: int (1-3, default 1) - Traversal depth

Used by: POST /api/v1/expand/entities
```

#### Output Models

**EntityAppearance**:
```
Entity mention in specific chunk.

Fields:
- vector_rowid: int - Chunk vector_rowid
- chunk_index: int - Chunk sequence number
- offset_start: int - Start position within chunk
- offset_end: int - End position within chunk

Used by: ExtractedEntity
```

**ExtractedEntity**:
```
Complete entity information with chunk mappings.

Fields:
- text: str - Original entity text
- normalized: str - Lowercase normalized form
- type_primary: str - Top-level type (e.g., "Framework")
- type_sub1: Optional[str] - Second-level type
- type_sub2: Optional[str] - Third-level type
- type_sub3: Optional[str] - Fourth-level type
- type_full: str - Full hierarchical type (e.g., "Framework::Backend::Python")
- confidence: float (0-1) - Extraction confidence
- neo4j_node_id: Optional[str] - Neo4j element ID
- context_before: str - Text before entity
- context_after: str - Text after entity
- sentence: str - Full sentence containing entity
- chunk_appearances: List[EntityAppearance] - All chunks containing entity
- spans_multiple_chunks: bool - True if in multiple chunks

Used by: IngestResponse
```

**ExtractedRelationship**:
```
Semantic relationship between entities.

Fields:
- subject_text: str - Subject entity text
- subject_neo4j_id: Optional[str] - Subject Neo4j node ID
- predicate: str - Relationship type (snake_case)
- object_text: str - Object entity text
- object_neo4j_id: Optional[str] - Object Neo4j node ID
- confidence: float (0-1) - Extraction confidence
- context: str - Supporting text from document
- neo4j_relationship_id: Optional[str] - Neo4j relationship ID
- spans_chunks: bool - True if entities in different chunks
- chunk_rowids: List[int] - All chunks involved

Used by: IngestResponse
```

**ProcessingSummary**:
```
Aggregated statistics from processing.

Fields:
- entities_by_type: Dict[str, int] - Entity counts by primary type
- relationships_by_predicate: Dict[str, int] - Relationship counts by predicate
- chunks_with_entities: int - Number of chunks containing entities
- avg_entities_per_chunk: float - Average entities per chunk

Used by: IngestResponse
```

**IngestResponse**:
```
Successful ingestion result.

Fields:
- success: bool - Processing status
- content_id: int - Source content ID
- neo4j_document_id: str - Neo4j Document node element ID
- entities_extracted: int - Total entity count
- relationships_extracted: int - Total relationship count
- processing_time_ms: int - Total processing time in milliseconds
- entities: List[ExtractedEntity] - Detailed entity data
- relationships: List[ExtractedRelationship] - Detailed relationship data
- summary: ProcessingSummary - Aggregated statistics

Returned by: POST /api/v1/ingest
```

**ErrorResponse**:
```
Structured error information.

Fields:
- success: bool (always False)
- error: str - Human-readable error message
- error_type: str - Exception class name
- content_id: Optional[int] - Content ID if available
- timestamp: str (ISO format) - Error timestamp

Returned by: All endpoints on error
```

**HealthStatus**:
```
Service health information.

Fields:
- status: str - "healthy" | "degraded" | "unhealthy"
- timestamp: datetime - Check timestamp
- services: Dict[str, str] - Dependency status (neo4j, vllm, gliner)
- version: str - Service version
- uptime_seconds: float - Time since startup

Returned by: GET /health
```

**ServiceStats**:
```
Processing metrics.

Fields:
- total_documents_processed: int
- total_entities_extracted: int
- total_relationships_extracted: int
- avg_processing_time_ms: float
- last_processed_at: Optional[datetime]
- queue_size: int (always 0, no queue in this service)
- failed_count: int

Returned by: GET /stats
```

#### Search Models

**EntityMatch**:
```
Matched entity from search.

Fields:
- entity_id: str - Neo4j element ID
- text: str - Entity text
- normalized: str - Normalized form
- type_primary: str - Primary type
- type_full: str - Full hierarchical type
- mention_count: int - Total mentions across documents
- confidence: float - Average extraction confidence

Returned by: POST /api/v1/search/entities
```

**ChunkMatch**:
```
Chunk containing entity mentions.

Fields:
- chunk_id: str - Neo4j Chunk node element ID
- vector_rowid: int - SQLite vector rowid (for retrieval)
- chunk_index: int - Chunk sequence in document
- entity_count: int - Number of matched entities in chunk
- matched_entities: List[str] - Entity names in chunk
- document_url: Optional[str] - Source document URL
- document_title: Optional[str] - Source document title

Returned by: POST /api/v1/search/chunks
```

**RelatedEntity**:
```
Entity discovered via expansion.

Fields:
- entity_id: str
- text: str
- normalized: str
- type_primary: str
- type_full: str
- mention_count: int
- relationship_type: Optional[str] - Relationship to original entity
- relationship_confidence: Optional[float]
- path_distance: int - Hops from starting entity

Returned by: POST /api/v1/expand/entities
```

## API Endpoints

### Document Ingestion

#### POST /api/v1/ingest

**Purpose**: Ingest document for entity and relationship extraction.

**Request**: IngestRequest
```json
{
  "content_id": 123,
  "url": "https://docs.example.com/page",
  "title": "Example Documentation",
  "markdown": "# Title\n\nContent with entities...",
  "chunks": [
    {
      "vector_rowid": 45001,
      "chunk_index": 0,
      "char_start": 0,
      "char_end": 2500,
      "text": "# Title\n\nFirst chunk text..."
    }
  ],
  "metadata": {
    "tags": "python,api",
    "timestamp": "2025-10-17T12:00:00Z"
  }
}
```

**Response**: IngestResponse (200 OK)
```json
{
  "success": true,
  "content_id": 123,
  "neo4j_document_id": "4:abc123:456",
  "entities_extracted": 42,
  "relationships_extracted": 18,
  "processing_time_ms": 8543,
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
      "neo4j_node_id": "4:def789:789",
      "context_before": "modern web ",
      "context_after": " for building",
      "sentence": "FastAPI is a modern web framework for building APIs.",
      "chunk_appearances": [
        {
          "vector_rowid": 45001,
          "chunk_index": 0,
          "offset_start": 120,
          "offset_end": 127
        }
      ],
      "spans_multiple_chunks": false
    }
  ],
  "relationships": [
    {
      "subject_text": "FastAPI",
      "subject_neo4j_id": "4:def789:789",
      "predicate": "uses",
      "object_text": "Pydantic",
      "object_neo4j_id": "4:def789:790",
      "confidence": 0.88,
      "context": "FastAPI uses Pydantic for data validation",
      "neo4j_relationship_id": "5:rel123:101",
      "spans_chunks": false,
      "chunk_rowids": [45001]
    }
  ],
  "summary": {
    "entities_by_type": {
      "Framework": 8,
      "Library": 12,
      "Language": 3
    },
    "relationships_by_predicate": {
      "uses": 10,
      "implements": 5,
      "based_on": 3
    },
    "chunks_with_entities": 15,
    "avg_entities_per_chunk": 2.8
  }
}
```

**Error Responses**:
- 422 Unprocessable Entity: Validation error (malformed request)
- 503 Service Unavailable: KG processor not initialized
- 500 Internal Server Error: Processing failure

**Processing Flow**:
1. Validate IngestRequest schema
2. Check KGProcessor initialization
3. Call processor.process_document() with request data
4. Update global processing statistics
5. Format entities and relationships for response
6. Return IngestResponse with all extracted knowledge

**Performance**: 10-15 seconds typical (2-3s entity extraction, 5-10s relationship extraction, 1-2s storage)

### Entity Search

#### POST /api/v1/search/entities

**Purpose**: Search for entities by text matching.

**Request**: EntitySearchRequest
```json
{
  "entity_terms": ["python", "fastapi"],
  "limit": 50,
  "min_mentions": 2
}
```

**Response**: EntitySearchResponse (200 OK)
```json
{
  "success": true,
  "entities": [
    {
      "entity_id": "4:abc:123",
      "text": "Python",
      "normalized": "python",
      "type_primary": "Language",
      "type_full": "Programming::Language",
      "mention_count": 147,
      "confidence": 0.92
    },
    {
      "entity_id": "4:abc:124",
      "text": "FastAPI",
      "normalized": "fastapi",
      "type_primary": "Framework",
      "type_full": "Framework::Backend::Python",
      "mention_count": 89,
      "confidence": 0.94
    }
  ],
  "total_found": 2
}
```

**Query Logic**:
- For each term: MATCH (e:Entity) WHERE toLower(e.text) CONTAINS toLower($term)
- Filter by min_mentions threshold
- Deduplicate across terms
- Order by mention_count DESC
- Limit results per term

**Use Case**: GraphRetriever finds entities matching user query terms

#### POST /api/v1/search/chunks

**Purpose**: Retrieve chunks containing specified entities.

**Request**: ChunkSearchRequest
```json
{
  "entity_names": ["Python", "FastAPI"],
  "limit": 100,
  "include_document_info": true
}
```

**Response**: ChunkSearchResponse (200 OK)
```json
{
  "success": true,
  "chunks": [
    {
      "chunk_id": "4:chunk:567",
      "vector_rowid": 45123,
      "chunk_index": 3,
      "entity_count": 2,
      "matched_entities": ["Python", "FastAPI"],
      "document_url": "https://docs.example.com/page",
      "document_title": "Example Documentation"
    }
  ],
  "total_found": 1
}
```

**Query Logic**:
- MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
- Filter by entity_ids or entity_names
- Group by chunk, collect matched entities
- Optionally join Document for URL/title
- Order by entity_count DESC (most relevant chunks first)

**Use Case**: GraphRetriever converts entity matches to vector_rowids for retrieval from SQLite

#### POST /api/v1/expand/entities

**Purpose**: Discover related entities via graph traversal.

**Request**: EntityExpansionRequest
```json
{
  "entity_names": ["FastAPI"],
  "max_expansions": 10,
  "min_confidence": 0.5,
  "expansion_depth": 1
}
```

**Response**: EntityExpansionResponse (200 OK)
```json
{
  "success": true,
  "original_entities": ["FastAPI"],
  "expanded_entities": [
    {
      "entity_id": "4:abc:125",
      "text": "Pydantic",
      "normalized": "pydantic",
      "type_primary": "Library",
      "type_full": "Programming::Library",
      "mention_count": 78,
      "relationship_type": "CO_OCCURS",
      "relationship_confidence": 0.9,
      "path_distance": 1
    },
    {
      "entity_id": "4:abc:126",
      "text": "Uvicorn",
      "normalized": "uvicorn",
      "type_primary": "Server",
      "type_full": "Web::Server",
      "mention_count": 45,
      "relationship_type": "CO_OCCURS",
      "relationship_confidence": 0.7,
      "path_distance": 1
    }
  ],
  "total_discovered": 2
}
```

**Query Logic**:
- MATCH (e1:Entity)-[:MENTIONED_IN]->(c:Chunk)<-[:MENTIONED_IN]-(e2:Entity)
- Filter by starting entity_names
- Count co-occurrences, assign confidence based on frequency
- Filter by min_confidence
- Order by co-occurrence count, then mention_count
- Limit to max_expansions

**Use Case**: EntityExpander enriches query with related terms for expanded retrieval

### Health & Monitoring

#### GET /health

**Purpose**: Check service health and dependency status.

**Response**: HealthStatus (200 OK)
```json
{
  "status": "healthy",
  "timestamp": "2025-10-17T14:32:18.234Z",
  "services": {
    "neo4j": "connected",
    "vllm": "connected (meta-llama/Llama-3.1-70B-Instruct)",
    "gliner": "loaded"
  },
  "version": "1.0.0",
  "uptime_seconds": 3600.5
}
```

**Status Values**:
- "healthy": All services connected/loaded
- "degraded": Some services unavailable but service functional
- "unhealthy": Critical service unavailable

**Checks Performed**:
1. Neo4j: Call driver.verify_connectivity()
2. vLLM: Call health_check() endpoint
3. GLiNER: Verify model loaded in memory

#### GET /stats

**Purpose**: Retrieve processing metrics.

**Response**: ServiceStats (200 OK)
```json
{
  "total_documents_processed": 523,
  "total_entities_extracted": 45234,
  "total_relationships_extracted": 12456,
  "avg_processing_time_ms": 8234.5,
  "last_processed_at": "2025-10-17T14:30:00.000Z",
  "queue_size": 0,
  "failed_count": 7
}
```

**Metrics Tracked**:
- Cumulative counts across all processed documents
- Rolling average of processing times
- Failure count (incremented on exception)

#### GET /

**Purpose**: Root endpoint with service information.

**Response**: (200 OK)
```json
{
  "service": "kg-service",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "health": "/health"
}
```

#### GET /api/v1/model-info

**Purpose**: Get information about loaded ML models.

**Response**: (200 OK)
```json
{
  "gliner": {
    "model": "urchade/gliner_large-v2.1",
    "threshold": 0.4,
    "status": "loaded",
    "entity_types_count": 300
  },
  "vllm": {
    "base_url": "http://localhost:8078",
    "model_name": "meta-llama/Llama-3.1-70B-Instruct",
    "status": "connected"
  }
}
```

## Interface Contracts

### External System Integration

**mcpragcrawl4ai → kg-service**:

1. **Document Ingestion**:
   - mcpragcrawl4ai crawls, cleans, chunks document
   - Generates embeddings, stores in SQLite content_vectors table
   - Calls POST /api/v1/ingest with full markdown and chunk boundaries
   - Receives entities/relationships with chunk mappings
   - Stores entity/relationship data in SQLite chunk_entities/chunk_relationships tables

2. **Entity Search**:
   - User submits query to mcpragcrawl4ai
   - mcpragcrawl4ai calls POST /api/v1/search/entities with query terms
   - Receives matching entities
   - Calls POST /api/v1/search/chunks with entity IDs
   - Gets vector_rowids for chunks containing entities
   - Retrieves chunk embeddings from SQLite for vector search

3. **Entity Expansion**:
   - mcpragcrawl4ai calls POST /api/v1/expand/entities with query entities
   - Receives related entities via co-occurrence
   - Expands query with related terms for broader retrieval

**kg-service → Neo4j**:
- All communication via Neo4j async driver
- Graph queries use Cypher
- Returns elementId (Neo4j unique identifier) for all nodes/relationships

**kg-service → vLLM**:
- HTTP POST to /v1/completions
- Request includes model name (auto-discovered), prompt, max_tokens, temperature
- Response contains generated text (JSON array of relationships)

**kg-service → GLiNER**:
- In-process model inference (no network call)
- Input: text string, entity_types list, threshold
- Output: list of predictions with text, label, score, start, end

## Error Handling

### Validation Errors (422)
```json
{
  "detail": [
    {
      "loc": ["body", "chunks", 0, "char_end"],
      "msg": "char_end must be greater than char_start",
      "type": "value_error"
    }
  ]
}
```

### Service Errors (500)
```json
{
  "success": false,
  "error": "Processing failed: vLLM request timeout",
  "error_type": "ModelUnavailableError",
  "content_id": 123,
  "timestamp": "2025-10-17T14:32:18.234Z"
}
```

### Not Found (404)
Returned when endpoint does not exist (standard FastAPI behavior).

### Service Unavailable (503)
Returned when KGProcessor not initialized or Neo4j unreachable.

## OpenAPI Documentation

FastAPI automatically generates OpenAPI spec at:
- `/docs`: Swagger UI (interactive)
- `/redoc`: ReDoc (alternative UI)
- `/openapi.json`: Raw OpenAPI 3.0 spec

All Pydantic models include schema examples for documentation.

---

[Next: Extractors Module Documentation](extractors.md)
