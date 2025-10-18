---
layout: default
title: System Architecture
---

# System Architecture

## Overview

The Knowledge Graph Service implements a layered architecture with clear separation between API handling, processing logic, ML inference, and data persistence. The system processes documents through a four-stage pipeline, transforming unstructured text into a structured knowledge graph.

## Architectural Layers

### Layer 1: API Interface
**Location**: `api/server.py`, `api/models.py`, `api/search_endpoints.py`

The API layer provides the external interface for document ingestion and knowledge retrieval:

- **FastAPI Application**: Async web server handling HTTP requests
- **Request Validation**: Pydantic models enforcing schema compliance
- **Response Formatting**: Structured JSON responses with entity and relationship data
- **Health Monitoring**: Status endpoints for operational visibility
- **Search Endpoints**: Entity search, chunk retrieval, and graph expansion

**Key Responsibilities**:
- HTTP request routing and validation
- Authentication and authorization (extensible)
- Error handling and exception translation
- Response serialization
- CORS middleware for cross-origin access

### Layer 2: Processing Pipeline
**Location**: `pipeline/processor.py`, `pipeline/chunk_mapper.py`

The pipeline layer orchestrates document processing workflows:

- **KGProcessor**: Main orchestrator coordinating extraction and storage
- **ChunkMapper**: Maps extracted entities/relationships to document chunk boundaries
- **Initialization Manager**: Handles service startup and component initialization
- **Statistics Generator**: Produces processing summaries and metrics

**Key Responsibilities**:
- Workflow coordination across extraction and storage stages
- Asynchronous task management
- Entity-to-chunk boundary mapping
- Cross-chunk relationship handling
- Processing metrics aggregation

### Layer 3: Extraction Engines
**Location**: `extractors/entity_extractor.py`, `extractors/relation_extractor.py`

The extraction layer interfaces with ML models for knowledge extraction:

- **EntityExtractor**: GLiNER-based entity recognition
- **RelationshipExtractor**: vLLM-powered semantic relationship extraction
- **Type Hierarchy Parser**: Processes hierarchical entity types
- **Context Extractor**: Captures surrounding text for entity mentions

**Key Responsibilities**:
- ML model inference execution
- Entity type classification and normalization
- Relationship identification and validation
- Confidence scoring
- Deduplication logic

### Layer 4: Storage & Persistence
**Location**: `storage/neo4j_client.py`, `storage/schema.py`

The storage layer manages graph database operations:

- **Neo4jClient**: Async driver wrapper for Neo4j operations
- **GraphSchema**: Schema definition and initialization
- **Node Creators**: Document, Chunk, and Entity node creation
- **Relationship Builders**: Semantic and structural relationship creation

**Key Responsibilities**:
- Neo4j connection management
- Schema initialization and validation
- Graph node and relationship CRUD operations
- Query optimization
- Transaction management

### Layer 5: External Clients
**Location**: `clients/vllm_client.py`

Client layer manages external service communication:

- **VLLMClient**: Async HTTP client for vLLM inference server
- **Model Discovery**: Automatic detection of available models
- **Retry Logic**: Exponential backoff on connection failures
- **Health Checks**: Service availability monitoring

**Key Responsibilities**:
- HTTP client lifecycle management
- Request formatting for vLLM API
- Response parsing and error handling
- Connection pooling
- Timeout management

### Layer 6: Configuration
**Location**: `config.py`

Configuration layer centralizes all service settings:

- **Settings**: Pydantic-based configuration with environment variable support
- **Validation**: Startup validation of critical settings
- **Logging**: Centralized logging configuration
- **Defaults**: Sensible defaults for all parameters

**Key Responsibilities**:
- Environment variable loading
- Configuration validation
- Default value management
- Settings access throughout application

## Data Flow Architecture

### Document Ingestion Flow

```
External System (mcpragcrawl4ai)
    |
    | POST /api/v1/ingest
    | - content_id
    | - url, title, markdown
    | - chunks[] with boundaries
    |
    v
FastAPI Server (api/server.py)
    |
    | IngestRequest validation
    |
    v
KGProcessor (pipeline/processor.py)
    |
    |---> EntityExtractor (extractors/entity_extractor.py)
    |     |
    |     | GLiNER Model Inference
    |     | - Split text if >1500 chars
    |     | - Extract entities with types
    |     | - Calculate confidence scores
    |     | - Capture context windows
    |     |
    |     v
    |     Entities[] with positions
    |
    |---> RelationshipExtractor (extractors/relation_extractor.py)
    |     |
    |     | vLLM LLM Inference
    |     | - Build extraction prompt
    |     | - Call vLLM completion API
    |     | - Parse JSON response
    |     | - Validate entity pairs
    |     |
    |     v
    |     Relationships[] with confidence
    |
    |---> ChunkMapper (pipeline/chunk_mapper.py)
    |     |
    |     | Boundary Calculation
    |     | - Map entity positions to chunks
    |     | - Calculate offsets within chunks
    |     | - Identify cross-chunk entities
    |     | - Determine relationship primary chunks
    |     |
    |     v
    |     Mapped Entities + Relationships
    |
    v
Neo4jClient (storage/neo4j_client.py)
    |
    | Graph Operations
    | 1. Create Document node
    | 2. Create Chunk nodes → HAS_CHUNK
    | 3. Create Entity nodes (MERGE on normalized)
    | 4. Create MENTIONED_IN relationships
    | 5. Create semantic relationships (USES, IMPLEMENTS, etc.)
    |
    v
Neo4j Graph Database
    |
    | Return node IDs
    |
    v
IngestResponse to External System
    - entities[] with chunk appearances
    - relationships[] with chunk mappings
    - neo4j_document_id
    - processing statistics
```

### Entity Search Flow

```
External System
    |
    | POST /api/v1/search/entities
    | - entity_terms[]
    | - limit, min_mentions
    |
    v
Search Endpoints (api/search_endpoints.py)
    |
    | For each search term:
    |
    v
Neo4jClient Query
    |
    | MATCH (e:Entity)
    | WHERE toLower(e.text) CONTAINS toLower($term)
    | RETURN entity details
    |
    v
EntityMatch[] Response
    - entity_id, text, normalized
    - type hierarchy
    - mention_count, confidence
```

### Chunk Retrieval Flow

```
External System
    |
    | POST /api/v1/search/chunks
    | - entity_ids[] OR entity_names[]
    |
    v
Search Endpoints
    |
    v
Neo4jClient Query
    |
    | MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
    | WHERE elementId(e) IN $entity_ids
    | OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c)
    | RETURN chunk details
    |
    v
ChunkMatch[] Response
    - chunk_id, vector_rowid
    - entity_count, matched_entities
    - document_url, document_title
```

### Entity Expansion Flow

```
External System
    |
    | POST /api/v1/expand/entities
    | - entity_names[]
    | - expansion_depth, max_expansions
    |
    v
Search Endpoints
    |
    v
Neo4jClient Query
    |
    | MATCH (e1:Entity)-[:MENTIONED_IN]->(c:Chunk)
    |       <-[:MENTIONED_IN]-(e2:Entity)
    | WHERE e1.text IN $entity_names
    | WITH e2, COUNT(DISTINCT c) as co_occurrence
    | RETURN related entities
    |
    v
RelatedEntity[] Response
    - entity details
    - relationship_type, confidence
    - path_distance
```

## Component Communication Patterns

### Synchronous Communication
- API → Pipeline: Direct async function calls
- Pipeline → Extractors: Direct async function calls
- Storage operations: Async/await pattern

### Asynchronous Communication
- vLLM Client: Async HTTP requests with timeout
- Neo4j Operations: Async driver with connection pooling

### Singleton Pattern
Global instances for shared resources:
- `get_entity_extractor()`: Single GLiNER model instance
- `get_relation_extractor()`: Single relationship extractor
- `get_vllm_client()`: Single HTTP client with connection pooling
- `get_neo4j_client()`: Single database driver
- `get_chunk_mapper()`: Stateless mapper instance

### Dependency Injection
Pipeline components receive dependencies via constructor injection, enabling:
- Unit testing with mock dependencies
- Configuration flexibility
- Clear dependency graphs

## Processing Pipeline Stages

### Stage 1: Entity Extraction
**Duration**: 2-3 seconds per document

1. **Text Preprocessing**: Split documents >1500 chars into GLiNER-compatible chunks
2. **Model Inference**: GLiNER predicts entities with types and confidence scores
3. **Type Parsing**: Hierarchical type strings split into primary/sub1/sub2/sub3
4. **Context Extraction**: Capture 50-char windows before/after each entity
5. **Deduplication**: Remove duplicate mentions at same position
6. **Position Tracking**: Record character start/end for each entity

**Output**: List of entities with text, normalized form, type hierarchy, confidence, context, and positions

### Stage 2: Relationship Extraction
**Duration**: 5-10 seconds per document

1. **Entity Preparation**: Convert entity dictionaries to EntityMention objects
2. **Prompt Construction**: Build LLM prompt with entity list and relationship types
3. **vLLM Inference**: Call completion API with guided JSON schema
4. **Response Parsing**: Extract JSON array from LLM response
5. **Validation**: Verify entity pairs exist and confidence meets threshold
6. **Enrichment**: Add entity positions and type information
7. **Deduplication**: Keep highest-confidence relationship for each (subject, predicate, object) triple

**Output**: List of relationships with subject/object entities, predicate, confidence, and context

### Stage 3: Chunk Mapping
**Duration**: <100ms per document

1. **Chunk Boundary Parsing**: Extract vector_rowid, chunk_index, char_start, char_end
2. **Entity-to-Chunk Mapping**: Calculate overlap between entity positions and chunk boundaries
3. **Offset Calculation**: Compute entity offsets within each chunk
4. **Multi-Chunk Detection**: Flag entities appearing in multiple chunks
5. **Relationship Chunk Mapping**: Determine chunks involved in each relationship
6. **Primary Chunk Selection**: Choose most relevant chunk for cross-chunk relationships

**Output**: Entities and relationships enriched with chunk appearance data

### Stage 4: Graph Storage
**Duration**: 1-2 seconds per document

1. **Document Node Creation**: MERGE on content_id
2. **Chunk Node Creation**: CREATE with vector_rowid as unique identifier
3. **Document-Chunk Linking**: CREATE HAS_CHUNK relationships
4. **Entity Node Creation**: MERGE on normalized text, increment mention_count
5. **Entity-Chunk Linking**: CREATE MENTIONED_IN with offset and context
6. **Semantic Relationship Creation**: CREATE dynamic relationship types (USES, IMPLEMENTS, etc.)
7. **Statistics Update**: Aggregate entity types and relationship counts

**Output**: Neo4j node IDs for all created/updated nodes

## Neo4j Graph Schema

### Node Labels

**Document Node**:
```
Properties:
- content_id (UNIQUE): Integer identifier from source system
- url: Document source URL
- title: Document title
- created_at: Timestamp of first ingestion
- updated_at: Timestamp of last update
```

**Chunk Node**:
```
Properties:
- vector_rowid (UNIQUE): SQLite vector table row ID
- chunk_index: Sequential chunk number in document
- char_start: Start position in full document
- char_end: End position in full document
- text_preview: First 200 characters
- created_at: Creation timestamp
```

**Entity Node**:
```
Properties:
- normalized (UNIQUE): Lowercase normalized entity text
- text: Original entity text
- type_primary: Top-level type (e.g., "Framework")
- type_sub1: Second-level type (e.g., "Backend")
- type_sub2: Third-level type (e.g., "Python")
- type_sub3: Fourth-level type (optional)
- type_full: Complete hierarchical type
- mention_count: Total mentions across all documents
- avg_confidence: Rolling average of extraction confidence
- created_at: First extraction timestamp
- updated_at: Last mention timestamp
```

### Relationship Types

**Structural Relationships**:

`HAS_CHUNK`: Document → Chunk
```
Properties: None (structural only)
```

`MENTIONED_IN`: Entity → Chunk
```
Properties:
- offset_start: Character offset in chunk
- offset_end: Character end offset in chunk
- confidence: Extraction confidence score
- context_before: Text before mention (100 chars)
- context_after: Text after mention (100 chars)
- sentence: Full sentence containing mention (500 chars)
- created_at: Creation timestamp
```

**Semantic Relationships** (Dynamic):

Dynamic relationship types based on extracted predicates (e.g., USES, IMPLEMENTS, EXTENDS):
```
Properties:
- confidence: Relationship confidence score
- context: Supporting text from document (500 chars)
- occurrence_count: Number of times relationship observed
- created_at: First observation timestamp
- updated_at: Last observation timestamp
```

### Indexes and Constraints

**Uniqueness Constraints**:
- `unique_document_content_id`: Ensures one Document node per content_id
- `unique_chunk_rowid`: Ensures one Chunk node per vector_rowid
- `unique_entity_normalized`: Ensures one Entity node per normalized text

**Performance Indexes**:
- `index_document_url`: Speed up document lookups by URL
- `index_entity_type_primary`: Filter entities by primary type
- `index_entity_type_full`: Filter entities by full hierarchical type
- `index_entity_text`: Full-text search on entity text
- `index_chunk_index`: Order chunks within documents

## Error Handling Strategy

### Validation Errors
- **Location**: API layer (Pydantic models)
- **Response**: 422 Unprocessable Entity with field-level errors
- **Recovery**: Client fixes request and resubmits

### Service Unavailability
- **vLLM Offline**: Return 503 Service Unavailable, log warning, skip relationship extraction
- **Neo4j Offline**: Fail fast during initialization, prevent service startup
- **GLiNER Load Failure**: Fatal error, service cannot start

### Processing Errors
- **Entity Extraction Failure**: Log error, return empty entity list, continue to next stage
- **Relationship Extraction Failure**: Log error, return empty relationship list
- **Chunk Mapping Failure**: Log error, skip chunk mapping, store entities without positions
- **Storage Failure**: Rollback transaction, return 500 Internal Server Error

### Retry Mechanism
- **vLLM Requests**: Exponential backoff with 3 retries, 30-second interval
- **Neo4j Operations**: Driver handles connection retry automatically
- **HTTP Timeouts**: Configurable per-endpoint (default 600s for vLLM)

## Scalability Considerations

### Horizontal Scaling
- **Stateless Design**: Each service instance operates independently
- **Shared Neo4j**: All instances connect to same Neo4j cluster
- **Load Balancing**: Standard HTTP load balancer distributes requests
- **Concurrent Processing**: Configure MAX_CONCURRENT_REQUESTS per instance

### Vertical Scaling
- **GLiNER Memory**: ~4GB per instance for model weights
- **CPU Utilization**: Multi-core CPU recommended for concurrent extraction
- **Neo4j Heap**: 2-4GB for production workloads
- **Connection Pools**: 50 Neo4j connections, 10 HTTP connections per instance

### Performance Optimization
- **Batch Processing**: Process multiple documents concurrently (up to 8)
- **Async I/O**: All network operations use async/await
- **Connection Pooling**: Reuse connections to Neo4j and vLLM
- **Schema Indexes**: Ensure all queries use indexed properties

### Bottleneck Analysis
- **Entity Extraction**: CPU-bound (GLiNER inference), parallelizable
- **Relationship Extraction**: Network-bound (vLLM latency), limited by vLLM throughput
- **Graph Storage**: I/O-bound (Neo4j writes), benefits from Neo4j clustering
- **Chunk Mapping**: CPU-bound but fast (<100ms), negligible impact

## Deployment Architecture

### Container Deployment
```
Docker Container 1: kg-service
- FastAPI application
- GLiNER model (loaded in memory)
- vLLM client (HTTP connection)
- Neo4j driver (connection pool)

Docker Container 2: neo4j-kg
- Neo4j 5.x database
- Graph data persistence
- APOC plugins

Docker Container 3: vllm-server (External)
- LLM model serving
- Inference API endpoint
```

### Environment Configuration
```
Service Discovery:
- NEO4J_URI: bolt://neo4j-kg:7687 (internal Docker network)
- VLLM_BASE_URL: http://vllm-server:8078 (external/internal)

Port Exposure:
- kg-service: 8088 (API endpoint)
- neo4j-kg: 7474 (browser), 7687 (bolt)
- vllm-server: 8078 (inference API)
```

### Production Topology
```
Load Balancer
    |
    |--- kg-service instance 1
    |--- kg-service instance 2
    |--- kg-service instance N
            |
            |--- Neo4j Cluster
            |    |--- Primary
            |    |--- Read Replica 1
            |    |--- Read Replica N
            |
            |--- vLLM Server (Shared)
```

## Monitoring and Observability

### Health Endpoints
- `GET /health`: Returns service status and dependency health
  - Overall status: healthy/degraded/unhealthy
  - Neo4j connectivity
  - vLLM availability
  - GLiNER model status
  - Service uptime

### Statistics Endpoints
- `GET /stats`: Returns processing metrics
  - Total documents processed
  - Total entities extracted
  - Total relationships extracted
  - Average processing time
  - Failed request count
  - Last processed timestamp

### Logging Strategy
- **Levels**: INFO for operations, DEBUG for detailed tracing, ERROR for failures
- **Rotation**: 10MB log files, 5 backups
- **Structured Format**: Timestamp, logger name, level, function, line, message
- **Key Events**:
  - Document ingestion start/complete
  - Entity extraction results
  - Relationship extraction results
  - Neo4j storage operations
  - vLLM request/response
  - Error conditions

### Performance Metrics
- **Request Timing**: X-Process-Time header on all responses
- **Stage Timing**: Logged for each pipeline stage
- **Model Performance**: Entity count, relationship count per document
- **Database Performance**: Neo4j query times (via driver logs)

## Security Considerations

### Authentication & Authorization
- **Current**: No authentication (internal service)
- **Extensible**: FastAPI middleware for JWT/OAuth integration
- **Recommendation**: Deploy behind API gateway with authentication

### Input Validation
- **Pydantic Models**: All requests validated against schema
- **Field Limits**: Max lengths on text fields (URLs, titles, markdown)
- **Type Enforcement**: Strict type checking on all inputs

### Neo4j Security
- **Credentials**: Loaded from environment variables
- **Connection**: TLS-encrypted Bolt protocol (configurable)
- **Access Control**: Database-level permissions on Neo4j user

### vLLM Security
- **No Authentication**: vLLM typically deployed internally without auth
- **Network Isolation**: Keep vLLM on internal network
- **Timeout Protection**: Prevent hanging requests

### Data Privacy
- **No PII Storage**: Only extracts technical entities/relationships
- **Document References**: Stores URLs and titles, not full content
- **Graph Isolation**: Each deployment has dedicated Neo4j database

---

[Next: API Layer Documentation](api.md)
