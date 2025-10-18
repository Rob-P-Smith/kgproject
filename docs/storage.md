---
layout: default
title: Storage Module
---

# Storage Module Documentation

## Module Overview

The storage module manages all interactions with the Neo4j graph database, including connection management, node/relationship creation, schema initialization, and query execution. It provides a clean abstraction layer between the processing pipeline and the graph database.

**Files**:
- `storage/neo4j_client.py`: Async Neo4j driver wrapper
- `storage/schema.py`: Graph schema definition and initialization
- `storage/__init__.py`: Module exports

## Neo4j Client

### Purpose

Provide async interface to Neo4j graph database for all CRUD operations on knowledge graph entities. Handles connection pooling, transaction management, and Cypher query execution.

### Architecture

**Class**: `Neo4jClient`

**Dependencies**:
- Neo4j async driver (official Python driver)
- Configuration settings (URI, credentials, connection params)

**Singleton Pattern**: Global instance via `get_neo4j_client()`

### Initialization

**Constructor**:
```python
def __init__(self):
    self.driver: Optional[AsyncDriver] = None
    self.uri = settings.NEO4J_URI
    self.user = settings.NEO4J_USER
    self.password = settings.NEO4J_PASSWORD
    self.database = settings.NEO4J_DATABASE
    self._connected = False
```

**Connection**:
```python
async def connect() -> bool:
    """
    Establish connection to Neo4j

    Creates async driver with:
    - URI: bolt://neo4j-kg:7687
    - Authentication: (neo4j, password)
    - Max connection lifetime: 3600 seconds
    - Connection pool size: 50
    - Connection timeout: 30 seconds

    Verifies connectivity before returning

    Returns:
        True if connected successfully, False otherwise
    """
```

**Error Handling**:
- `ServiceUnavailable`: Neo4j server not reachable
- `AuthError`: Invalid credentials
- Both exceptions logged and return False

### Node Creation

#### Create Document Node

**Method**: `create_document(content_id, url, title, metadata) -> str`

**Purpose**: Create or update Document node representing source document

**Cypher Query**:
```cypher
MERGE (d:Document {content_id: $content_id})
SET d.url = $url,
    d.title = $title,
    d.created_at = COALESCE(d.created_at, datetime()),
    d.updated_at = datetime()
RETURN elementId(d) AS node_id
```

**Behavior**:
- **On first ingestion**: Creates new Document node with created_at timestamp
- **On re-ingestion**: Updates url, title, updated_at (preserves created_at)
- **Idempotent**: Safe to call multiple times with same content_id

**Parameters**:
- `content_id`: int - Unique identifier from source system (serves as UNIQUE constraint)
- `url`: str - Document source URL
- `title`: str - Document title
- `metadata`: Dict - NOT stored in Neo4j (graph DB doesn't store document content)

**Returns**: Neo4j elementId (string like "4:abc123:456")

**Use Case**: Called once per document during ingestion

#### Create Chunk Node

**Method**: `create_chunk(document_node_id, vector_rowid, chunk_index, char_start, char_end, text_preview) -> str`

**Purpose**: Create Chunk node and link to parent Document

**Cypher Query**:
```cypher
MATCH (d:Document)
WHERE elementId(d) = $doc_id
MERGE (c:Chunk {vector_rowid: $vector_rowid})
SET c.chunk_index = $chunk_index,
    c.char_start = $char_start,
    c.char_end = $char_end,
    c.text_preview = $text_preview,
    c.created_at = COALESCE(c.created_at, datetime())
MERGE (d)-[:HAS_CHUNK]->(c)
RETURN elementId(c) AS node_id
```

**Behavior**:
- **MERGE on vector_rowid**: Ensures one Chunk node per SQLite vector table row
- **Links to Document**: Creates HAS_CHUNK relationship
- **Updates properties**: Refreshes chunk metadata on re-ingestion

**Parameters**:
- `document_node_id`: str - Neo4j elementId of parent Document
- `vector_rowid`: int - SQLite content_vectors rowid (serves as UNIQUE constraint)
- `chunk_index`: int - Sequential chunk number in document (0-indexed)
- `char_start`: int - Start position in full document
- `char_end`: int - End position in full document
- `text_preview`: str - First 200 characters of chunk (for debugging/visualization)

**Returns**: Neo4j elementId of Chunk node

**Use Case**: Called once per chunk during ingestion

#### Create Entity Node

**Method**: `create_entity(text, normalized, type_primary, type_sub1, type_sub2, type_sub3, type_full, confidence) -> str`

**Purpose**: Create or update Entity node, incrementing mention count on duplicates

**Cypher Query**:
```cypher
MERGE (e:Entity {normalized: $normalized})
ON CREATE SET
    e.text = $text,
    e.type_primary = $type_primary,
    e.type_sub1 = $type_sub1,
    e.type_sub2 = $type_sub2,
    e.type_sub3 = $type_sub3,
    e.type_full = $type_full,
    e.created_at = datetime(),
    e.mention_count = 1,
    e.avg_confidence = $confidence
ON MATCH SET
    e.mention_count = e.mention_count + 1,
    e.avg_confidence = (e.avg_confidence * (e.mention_count - 1) + $confidence) / e.mention_count,
    e.updated_at = datetime()
RETURN elementId(e) AS node_id
```

**Behavior**:
- **ON CREATE**: First mention of entity, set all properties
- **ON MATCH**: Subsequent mentions, increment count and update rolling average confidence
- **Normalization**: Entities merged on lowercase normalized text (e.g., "FastAPI" and "fastapi" → same node)

**Parameters**:
- `text`: str - Original entity text (e.g., "FastAPI")
- `normalized`: str - Lowercase normalized form (e.g., "fastapi") - UNIQUE constraint key
- `type_primary`: str - Top-level type (e.g., "Framework")
- `type_sub1`: Optional[str] - Second-level type (e.g., "Backend")
- `type_sub2`: Optional[str] - Third-level type (e.g., "Python")
- `type_sub3`: Optional[str] - Fourth-level type (optional)
- `type_full`: str - Complete hierarchical type (e.g., "Framework::Backend::Python")
- `confidence`: float - Extraction confidence (0-1)

**Returns**: Neo4j elementId of Entity node

**Rolling Average Calculation**:
```
new_avg = (old_avg * (count - 1) + new_confidence) / count
```

**Use Case**: Called once per unique entity during ingestion

### Relationship Creation

#### Link Entity to Chunk

**Method**: `link_entity_to_chunk(entity_node_id, chunk_node_id, offset_start, offset_end, confidence, context_before, context_after, sentence)`

**Purpose**: Create MENTIONED_IN relationship between Entity and Chunk with mention metadata

**Cypher Query**:
```cypher
MATCH (e:Entity), (c:Chunk)
WHERE elementId(e) = $entity_id AND elementId(c) = $chunk_id
MERGE (e)-[m:MENTIONED_IN]->(c)
SET m.offset_start = $offset_start,
    m.offset_end = $offset_end,
    m.confidence = $confidence,
    m.context_before = $context_before,
    m.context_after = $context_after,
    m.sentence = $sentence,
    m.created_at = COALESCE(m.created_at, datetime())
```

**Behavior**:
- **MERGE relationship**: Ensures single MENTIONED_IN per entity-chunk pair
- **Updates properties**: Refreshes mention metadata on re-ingestion
- **Stores context**: Preserves surrounding text for retrieval

**Parameters**:
- `entity_node_id`: str - Entity elementId
- `chunk_node_id`: str - Chunk elementId
- `offset_start`: int - Character offset in chunk where entity starts
- `offset_end`: int - Character offset in chunk where entity ends
- `confidence`: float - Mention confidence
- `context_before`: str - Up to 100 chars before mention
- `context_after`: str - Up to 100 chars after mention
- `sentence`: str - Up to 500 chars of sentence containing mention

**Returns**: None (no return value)

**Use Case**: Called once per entity appearance in each chunk

#### Create Semantic Relationship

**Method**: `create_relationship(subject_normalized, predicate, object_normalized, confidence, context)`

**Purpose**: Create semantic relationship between two entities (e.g., FastAPI USES Pydantic)

**Cypher Query** (dynamic relationship type):
```cypher
MATCH (s:Entity {normalized: $subject})
MATCH (o:Entity {normalized: $object})
MERGE (s)-[r:USES]->(o)  # Relationship type from predicate
ON CREATE SET
    r.confidence = $confidence,
    r.context = $context,
    r.created_at = datetime(),
    r.occurrence_count = 1
ON MATCH SET
    r.confidence = (r.confidence * r.occurrence_count + $confidence) / (r.occurrence_count + 1),
    r.occurrence_count = r.occurrence_count + 1,
    r.updated_at = datetime()
```

**Dynamic Relationship Types**:
- Predicate converted to uppercase with underscores: "uses" → "USES", "based_on" → "BASED_ON"
- Neo4j relationship types created dynamically at runtime
- Examples: USES, IMPLEMENTS, EXTENDS, BASED_ON, COMPETES_WITH, etc.

**Behavior**:
- **ON CREATE**: First observation of relationship
- **ON MATCH**: Subsequent observations, increment count and update rolling average
- **Directional**: Relationships have direction (subject → object)

**Parameters**:
- `subject_normalized`: str - Subject entity normalized name
- `predicate`: str - Relationship type (snake_case)
- `object_normalized`: str - Object entity normalized name
- `confidence`: float - Relationship confidence
- `context`: str - Supporting text (up to 500 chars)

**Returns**: None

**Use Case**: Called once per extracted relationship during ingestion

#### Update Co-occurrence

**Method**: `update_co_occurrence(entity1_normalized, entity2_normalized, chunk_rowid)`

**Purpose**: Track entity co-occurrence in chunks (currently DISABLED in pipeline)

**Cypher Query**:
```cypher
MATCH (e1:Entity {normalized: $entity1})
MATCH (e2:Entity {normalized: $entity2})
WHERE e1.normalized < e2.normalized  -- Ensure consistent direction
MERGE (e1)-[co:CO_OCCURS_WITH]->(e2)
ON CREATE SET
    co.count = 1,
    co.chunk_rowids = [$chunk_rowid],
    co.created_at = datetime()
ON MATCH SET
    co.count = co.count + 1,
    co.chunk_rowids = co.chunk_rowids + $chunk_rowid,
    co.updated_at = datetime()
```

**Behavior**:
- **Consistent Direction**: Ensures e1.normalized < e2.normalized (alphabetical order)
- **Accumulates Chunks**: Tracks all chunks where entities co-occur
- **Increments Count**: Counts total co-occurrences across all documents

**Note**: Currently disabled in pipeline (`_update_co_occurrences` not called) to avoid creating too many relationships. Co-occurrence analysis done via search endpoints using MENTIONED_IN traversal instead.

### Query Operations

#### Get Document Stats

**Method**: `get_document_stats(content_id) -> Dict`

**Purpose**: Retrieve statistics for a specific document

**Cypher Query**:
```cypher
MATCH (d:Document {content_id: $content_id})
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(c)
RETURN
    elementId(d) AS doc_id,
    d.url AS url,
    d.title AS title,
    COUNT(DISTINCT c) AS chunk_count,
    COUNT(DISTINCT e) AS entity_count
```

**Returns**:
```python
{
  "doc_id": "4:abc:123",
  "url": "https://example.com/doc",
  "title": "Example Document",
  "chunk_count": 18,
  "entity_count": 42
}
```

**Use Case**: Monitoring, debugging, statistics endpoints

### Health Check

**Method**: `health_check() -> Dict`

**Purpose**: Verify database connectivity and return status

**Query**: `RETURN 1 AS health`

**Returns**:
```python
{
  "status": "connected",
  "uri": "bolt://neo4j-kg:7687",
  "database": "neo4j"
}
# or on error:
{
  "status": "error",
  "message": "Connection timeout"
}
```

**Use Case**: Health endpoint at API layer

### Connection Management

**Context Manager Support**:
```python
async with Neo4jClient() as client:
    # Automatically connects on enter
    # Automatically closes on exit
    pass
```

**Explicit Lifecycle**:
```python
client = Neo4jClient()
await client.connect()
# ... operations ...
await client.close()
```

**Connection Pooling**:
- Driver maintains connection pool (max 50 connections)
- Connections reused across transactions
- Automatic reconnection on transient failures

## Graph Schema

### Purpose

Define and initialize the knowledge graph schema, including node labels, relationship types, uniqueness constraints, and performance indexes.

### Architecture

**Class**: `GraphSchema`

**Dependencies**:
- Neo4jClient (connected instance)

**Usage**:
```python
schema = GraphSchema(neo4j_client)
results = await schema.initialize_schema()
```

### Schema Definition

#### Node Labels

**Document**:
```
Label: Document
Purpose: Represents source document from external system
Unique Constraint: content_id
Indexed Properties: url
```

**Chunk**:
```
Label: Chunk
Purpose: Represents vector-embedded text chunk
Unique Constraint: vector_rowid
Indexed Properties: chunk_index
Links to: SQLite content_vectors table via vector_rowid
```

**Entity**:
```
Label: Entity
Purpose: Represents named entity extracted from text
Unique Constraint: normalized
Indexed Properties: type_primary, type_full, text
```

#### Relationship Types

**Structural Relationships**:

`HAS_CHUNK`: Document → Chunk
```
Purpose: Links document to its chunks
Properties: None
Cardinality: One-to-many
```

`MENTIONED_IN`: Entity → Chunk
```
Purpose: Links entity to chunks where it appears
Properties:
  - offset_start: int
  - offset_end: int
  - confidence: float
  - context_before: str (100 chars)
  - context_after: str (100 chars)
  - sentence: str (500 chars)
  - created_at: datetime
Cardinality: Many-to-many
```

**Semantic Relationships** (Dynamic):

Dynamic types created from extracted predicates:
- USES, IMPLEMENTS, EXTENDS, DEPENDS_ON, REQUIRES
- PROVIDES, SUPPORTS, INTEGRATES_WITH, BASED_ON
- BUILT_WITH, POWERED_BY, RUNS_ON, COMPATIBLE_WITH
- SIMILAR_TO, ALTERNATIVE_TO, COMPETES_WITH, DIFFERS_FROM
- REPLACES, SUPERSEDES, EVOLVED_FROM
- PART_OF, CONTAINS, INCLUDES, COMPOSED_OF
- CATEGORY_OF, TYPE_OF, INSTANCE_OF, SUBCLASS_OF
- PROCESSES, GENERATES, TRANSFORMS, ANALYZES
- VALIDATES, HANDLES, MANAGES, CONTROLS
- DEVELOPED_BY, MAINTAINED_BY, CREATED_BY, DESIGNED_BY
- DOCUMENTED_IN, DESCRIBED_IN, DEFINED_IN, REFERENCED_IN
- CONFIGURED_WITH, SETTINGS_FOR, PARAMETER_OF, OPTION_FOR
- OPTIMIZES, IMPROVES, ACCELERATES, SCALES_WITH

```
Purpose: Semantic relationships between entities
Properties:
  - confidence: float
  - context: str (500 chars)
  - occurrence_count: int
  - created_at: datetime
  - updated_at: datetime
Cardinality: Many-to-many
Direction: Subject → Object
```

`CO_OCCURS_WITH`: Entity → Entity (optional, currently unused)
```
Purpose: Tracks entity co-occurrence in chunks
Properties:
  - count: int
  - chunk_rowids: List[int]
  - created_at: datetime
  - updated_at: datetime
Cardinality: Many-to-many
Direction: Alphabetically ordered (e1.normalized < e2.normalized)
```

### Schema Initialization

**Method**: `initialize_schema() -> Dict`

**Purpose**: Create constraints and indexes on startup

**Process**:

1. **Create Uniqueness Constraints**:
```cypher
CREATE CONSTRAINT unique_document_content_id IF NOT EXISTS
FOR (d:Document) REQUIRE d.content_id IS UNIQUE

CREATE CONSTRAINT unique_chunk_rowid IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.vector_rowid IS UNIQUE

CREATE CONSTRAINT unique_entity_normalized IF NOT EXISTS
FOR (e:Entity) REQUIRE e.normalized IS UNIQUE
```

2. **Create Performance Indexes**:
```cypher
CREATE INDEX index_document_url IF NOT EXISTS
FOR (d:Document) ON (d.url)

CREATE INDEX index_entity_type_primary IF NOT EXISTS
FOR (e:Entity) ON (e.type_primary)

CREATE INDEX index_entity_type_full IF NOT EXISTS
FOR (e:Entity) ON (e.type_full)

CREATE INDEX index_entity_text IF NOT EXISTS
FOR (e:Entity) ON (e.text)

CREATE INDEX index_chunk_index IF NOT EXISTS
FOR (c:Chunk) ON (c.chunk_index)
```

**Returns**:
```python
{
  "constraints_created": 3,
  "indexes_created": 5,
  "errors": []
}
```

**Error Handling**:
- Constraints/indexes already existing: Logged as warning, not counted as error
- Creation failures: Logged, added to errors list
- Service continues even if some constraints/indexes fail

**Timing**: ~500ms on first run, <50ms on subsequent (already exists)

### Schema Utilities

**Method**: `get_schema_info() -> Dict`

Returns current schema metadata:
```python
{
  "constraints": [
    {"name": "unique_document_content_id", "type": "UNIQUENESS", "entityType": "NODE"},
    ...
  ],
  "indexes": [
    {"name": "index_entity_text", "type": "RANGE", "entityType": "NODE", "properties": ["text"]},
    ...
  ],
  "node_counts": {
    "Document": 523,
    "Chunk": 9414,
    "Entity": 12456
  },
  "relationship_counts": {
    "HAS_CHUNK": 9414,
    "MENTIONED_IN": 67823,
    "USES": 3421,
    "IMPLEMENTS": 1234,
    ...
  }
}
```

**Method**: `validate_schema() -> Dict`

Checks schema integrity:
```python
{
  "valid": True,
  "issues": [
    "Found 5 entities with no chunk mentions (may be expected for entity normalization)"
  ]
}
```

**Validation Checks**:
- Orphaned chunks (no parent Document)
- Entities with no mentions (may occur due to normalization)

**Method**: `clear_all_data() -> int`

**Warning**: DANGEROUS - deletes all data

```cypher
MATCH (n)
DETACH DELETE n
RETURN count(n) AS deleted
```

**Use Case**: Testing, development, data reset

## Storage Patterns

### Entity Deduplication

**Pattern**: MERGE on normalized text
```
Entity "FastAPI" in doc1 → normalized="fastapi"
Entity "fastapi" in doc2 → normalized="fastapi"
Result: Same Entity node, mention_count=2
```

**Benefits**:
- Consistent entity representation across documents
- Accurate mention counting
- Simplified querying (one node per entity)

### Relationship Aggregation

**Pattern**: MERGE on (subject, predicate, object), increment occurrence_count
```
"FastAPI uses Pydantic" in doc1 → confidence=0.9, occurrence_count=1
"FastAPI uses Pydantic" in doc2 → confidence=0.85, occurrence_count=2, avg_confidence=0.875
```

**Benefits**:
- Avoid duplicate relationships
- Track relationship strength across documents
- Rolling average confidence

### Chunk-Entity Linking

**Pattern**: MENTIONED_IN relationship with offset metadata
```
Entity "FastAPI" appears in Chunk 45001 at offset 120-127
→ (Entity)-[MENTIONED_IN {offset_start: 120, offset_end: 127}]->(Chunk)
```

**Benefits**:
- Precise retrieval (know exact position in chunk)
- Context preservation (sentence, surrounding text)
- Multi-mention support (entity can appear in multiple chunks)

### Document-Chunk Hierarchy

**Pattern**: HAS_CHUNK relationship preserves document structure
```
Document → Chunk[0], Chunk[1], ..., Chunk[N]
Each chunk retains char_start, char_end for reconstruction
```

**Benefits**:
- Reconstruct document from chunks
- Maintain chunk order
- Support chunk-level retrieval with document context

## Performance Considerations

### Query Optimization

**Indexed Lookups**:
- Entity search by text: Uses index_entity_text
- Entity search by type: Uses index_entity_type_primary
- Document lookup by content_id: Uses unique_document_content_id (automatic index)
- Chunk lookup by vector_rowid: Uses unique_chunk_rowid (automatic index)

**Traversal Efficiency**:
- Entity → Chunk: Direct MENTIONED_IN traversal
- Chunk → Document: Direct HAS_CHUNK (reverse) traversal
- Entity → Co-occurring Entity: Two-hop via MENTIONED_IN

**Query Patterns**:
```cypher
-- Fast: Uses constraint index
MATCH (e:Entity {normalized: "fastapi"})

-- Fast: Uses type index
MATCH (e:Entity {type_primary: "Framework"})

-- Fast: Uses constraint + single hop
MATCH (e:Entity {normalized: "fastapi"})-[:MENTIONED_IN]->(c:Chunk)

-- Medium: Two hops, but indexed start
MATCH (e1:Entity {normalized: "fastapi"})-[:MENTIONED_IN]->(c:Chunk)<-[:MENTIONED_IN]-(e2:Entity)

-- Slow: Full graph scan (avoid in production)
MATCH (e:Entity)-[r]-(other)
RETURN e, r, other
```

### Write Performance

**Batch Writes**:
- Current implementation: Sequential writes per entity/chunk
- Optimization opportunity: Batch CREATE statements via UNWIND

**Transaction Overhead**:
- Each method call = separate transaction (auto-commit)
- Optimization opportunity: Single transaction for entire document

**MERGE Performance**:
- MERGE requires index lookup + potential write
- Faster than separate MATCH + CREATE logic
- Benefits from uniqueness constraints

### Scalability

**Node Growth**:
- Linear growth with documents: O(D * C) chunks (D=documents, C=chunks/doc)
- Linear growth with vocabulary: O(V) entities (V=unique entities across corpus)
- Quadratic growth potential: O(V²) relationships (bounded by actual relationships)

**Index Maintenance**:
- Automatic index updates on writes
- Index size grows with node count
- Recommend periodic index rebuilds for large deployments

**Connection Pooling**:
- 50 concurrent connections supported
- Reuse connections across requests
- No connection exhaustion under normal load

---

[Next: Configuration & Clients Documentation](configuration.md)
