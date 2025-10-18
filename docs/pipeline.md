---
layout: default
title: Pipeline Module
---

# Pipeline Module Documentation

## Module Overview

The pipeline module orchestrates the complete knowledge graph processing workflow, coordinating entity extraction, relationship extraction, chunk mapping, and graph storage. It serves as the central coordinator between the API layer and the specialized processing components.

**Files**:
- `pipeline/processor.py`: Main processing orchestrator (KGProcessor)
- `pipeline/chunk_mapper.py`: Entity/relationship to chunk mapping
- `pipeline/__init__.py`: Module exports

## KG Processor

### Purpose

Coordinate the complete document processing pipeline from raw markdown to stored knowledge graph. Manages component initialization, workflow execution, error handling, and result aggregation.

### Architecture

**Class**: `KGProcessor`

**Dependencies**:
- EntityExtractor: GLiNER-based entity extraction
- RelationshipExtractor: vLLM-based relationship extraction
- ChunkMapper: Boundary mapping logic
- Neo4jClient: Graph database operations

**Singleton Pattern**: Single global instance shared across all requests

### Initialization

**Constructor**:
```python
def __init__(self):
    self.entity_extractor = get_entity_extractor()
    self.relation_extractor = get_relation_extractor()
    self.chunk_mapper = get_chunk_mapper()
    self.neo4j_client = get_neo4j_client()
    self._initialized = False
    self._init_lock = asyncio.Lock()
```

**Async Initialization**:
```python
async def initialize() -> bool:
    """
    Initialize all components

    Steps:
    1. Acquire initialization lock (prevent concurrent init)
    2. Check if already initialized (idempotent)
    3. Connect to Neo4j database
    4. Initialize graph schema (constraints, indexes)
    5. Set _initialized flag
    6. Release lock

    Returns:
        True if successful, False if Neo4j connection failed
    """
```

**Schema Initialization Results**:
```python
{
  "constraints_created": 3,  # unique_document_content_id, unique_chunk_rowid, unique_entity_normalized
  "indexes_created": 5,      # indexes on url, type_primary, type_full, text, chunk_index
  "errors": []               # Any non-fatal errors during creation
}
```

**Initialization Timing**: ~1-2 seconds (Neo4j connection + schema creation)

### Processing Pipeline

**Method**: `process_document(content_id, url, title, markdown, chunks, metadata) -> Dict`

**Purpose**: Execute complete pipeline for single document

**Input Parameters**:
- `content_id`: int - Unique identifier from source system
- `url`: str - Document source URL
- `title`: str - Document title
- `markdown`: str - Full document text
- `chunks`: List[Dict] - Chunk boundaries from source system
- `metadata`: Dict - Optional metadata (not stored in Neo4j)

**Output**:
```python
{
  "success": True,
  "content_id": 123,
  "neo4j_document_id": "4:abc123:456",
  "entities_extracted": 42,
  "relationships_extracted": 18,
  "processing_time_ms": 8543.21,
  "entities": [ExtractedEntity, ...],
  "relationships": [ExtractedRelationship, ...],
  "summary": ProcessingSummary
}
```

**Processing Stages**:

#### Stage 1: Entity Extraction (2-3 seconds)

```python
logger.info("🔍 Step 1: Extracting entities with GLiNER...")
entities = self.entity_extractor.extract(markdown)
logger.info(f"✅ ENTITIES EXTRACTED: {len(entities)} entities found")

# entities = [
#   {
#     "text": "FastAPI",
#     "normalized": "fastapi",
#     "type_full": "Framework::Backend::Python",
#     "type_primary": "Framework",
#     "type_sub1": "Backend",
#     "type_sub2": "Python",
#     "type_sub3": None,
#     "confidence": 0.95,
#     "start": 120,
#     "end": 127,
#     "context_before": "...",
#     "context_after": "...",
#     "sentence": "...",
#     "extraction_method": "gliner"
#   },
#   ...
# ]
```

**GLiNER Processing**:
- Automatically chunks text >1500 chars
- Extracts entities with hierarchical types
- Captures context windows and sentences
- Returns entities with document positions

#### Stage 2: Relationship Extraction (5-10 seconds)

```python
logger.info("🔗 Step 2: Extracting relationships with vLLM...")
relationships = await self.relation_extractor.extract_relationships(
    markdown,
    entities
)
logger.info(f"✅ RELATIONSHIPS EXTRACTED: {len(relationships)} relationships found")

# relationships = [
#   {
#     "subject_text": "FastAPI",
#     "subject_normalized": "fastapi",
#     "subject_type": "Framework::Backend::Python",
#     "predicate": "uses",
#     "object_text": "Pydantic",
#     "object_normalized": "pydantic",
#     "object_type": "Programming::Library",
#     "confidence": 0.88,
#     "context": "FastAPI uses Pydantic for data validation",
#     "subject_start": 120,
#     "subject_end": 127,
#     "object_start": 145,
#     "object_end": 153
#   },
#   ...
# ]
```

**vLLM Processing**:
- Builds prompt with entity list
- Calls LLM to identify relationships
- Parses JSON response
- Validates entity pairs
- Filters by confidence threshold

#### Stage 3: Chunk Mapping (<100ms)

```python
logger.info("Step 3: Mapping entities and relationships to chunks...")

# Map entities to chunk boundaries
mapped_entities = self.chunk_mapper.map_entities_to_chunks(
    entities,
    chunks
)

# Map relationships to chunks
mapped_relationships = self.chunk_mapper.map_relationships_to_chunks(
    relationships,
    mapped_entities,
    chunks
)

logger.info("✓ Chunk mapping complete")

# mapped_entities = [
#   {
#     ...entity fields...,
#     "chunk_appearances": [
#       {
#         "vector_rowid": 45001,
#         "chunk_index": 0,
#         "offset_start": 120,
#         "offset_end": 127
#       }
#     ],
#     "spans_multiple_chunks": False,
#     "num_chunks": 1
#   },
#   ...
# ]

# mapped_relationships = [
#   {
#     ...relationship fields...,
#     "spans_chunks": False,
#     "chunk_rowids": [45001],
#     "primary_chunk_rowid": 45001,
#     "num_chunks_involved": 1
#   },
#   ...
# ]
```

**Chunk Mapping Logic**:
- Calculates overlap between entity positions and chunk boundaries
- Computes offsets within each chunk
- Identifies cross-chunk entities
- Determines primary chunk for relationships

#### Stage 4: Neo4j Storage (1-2 seconds)

```python
logger.info("💾 Step 4: Storing in Neo4j...")

neo4j_result = await self._store_in_neo4j(
    content_id=content_id,
    url=url,
    title=title,
    metadata=metadata,
    chunks=chunks,
    entities=mapped_entities,
    relationships=mapped_relationships
)

logger.info(f"✅ NEO4J STORAGE COMPLETE:")
logger.info(f"   - Document node: {neo4j_result['document_node_id']}")
logger.info(f"   - Chunks stored: {neo4j_result['chunk_count']}")
logger.info(f"   - Entities stored: {neo4j_result['entity_count']}")
logger.info(f"   - Relationships stored: {neo4j_result['relationship_count']}")

# neo4j_result = {
#   "document_node_id": "4:abc:123",
#   "chunk_count": 18,
#   "entity_count": 42,
#   "relationship_count": 18
# }
```

**Neo4j Storage Operations**:
1. Create Document node (MERGE on content_id)
2. Create Chunk nodes and link to Document
3. Create Entity nodes (MERGE on normalized text)
4. Link entities to chunks (MENTIONED_IN relationships)
5. Create semantic relationships between entities
6. Return node IDs

#### Stage 5: Response Formatting

```python
# Generate summary statistics
summary = self.chunk_mapper.generate_mapping_summary(
    mapped_entities,
    mapped_relationships,
    chunks
)

# Format entities for API response
formatted_entities = self._format_entities_for_response(mapped_entities)

# Format relationships for API response
formatted_relationships = self._format_relationships_for_response(mapped_relationships)

# Build result
result = {
    "success": True,
    "content_id": content_id,
    "neo4j_document_id": neo4j_result["document_node_id"],
    "entities_extracted": len(mapped_entities),
    "relationships_extracted": len(mapped_relationships),
    "processing_time_ms": processing_time,
    "entities": formatted_entities,
    "relationships": formatted_relationships,
    "summary": summary
}
```

### Neo4j Storage Implementation

**Method**: `_store_in_neo4j(...) -> Dict`

**Storage Workflow**:

1. **Create Document Node**:
```python
doc_node_id = await self.neo4j_client.create_document(
    content_id=content_id,
    url=url,
    title=title,
    metadata=metadata
)
# MERGE (d:Document {content_id: $content_id})
# SET d.url, d.title, d.updated_at
# ON CREATE SET d.created_at
```

2. **Create Chunk Nodes**:
```python
chunk_node_map = {}  # vector_rowid -> node_id

for chunk in chunks:
    chunk_node_id = await self.neo4j_client.create_chunk(
        document_node_id=doc_node_id,
        vector_rowid=chunk["vector_rowid"],
        chunk_index=chunk["chunk_index"],
        char_start=chunk["char_start"],
        char_end=chunk["char_end"],
        text_preview=chunk["text"][:200]
    )
    chunk_node_map[chunk["vector_rowid"]] = chunk_node_id

# MERGE (c:Chunk {vector_rowid: $vector_rowid})
# SET c.chunk_index, c.char_start, c.char_end, c.text_preview
# MERGE (d)-[:HAS_CHUNK]->(c)
```

3. **Create Entity Nodes**:
```python
entity_node_map = {}  # normalized -> node_id

for entity in entities:
    entity_node_id = await self.neo4j_client.create_entity(
        text=entity["text"],
        normalized=entity["normalized"],
        type_primary=entity["type_primary"],
        type_sub1=entity.get("type_sub1"),
        type_sub2=entity.get("type_sub2"),
        type_sub3=entity.get("type_sub3"),
        type_full=entity["type_full"],
        confidence=entity["confidence"]
    )
    entity_node_map[entity["normalized"]] = entity_node_id

# MERGE (e:Entity {normalized: $normalized})
# ON CREATE SET e.text, e.type_*, e.mention_count = 1, e.avg_confidence
# ON MATCH SET e.mention_count += 1, e.avg_confidence = rolling average
```

4. **Link Entities to Chunks**:
```python
for entity in entities:
    entity_node_id = entity_node_map[entity["normalized"]]

    for appearance in entity["chunk_appearances"]:
        chunk_node_id = chunk_node_map[appearance["vector_rowid"]]

        await self.neo4j_client.link_entity_to_chunk(
            entity_node_id=entity_node_id,
            chunk_node_id=chunk_node_id,
            offset_start=appearance["offset_start"],
            offset_end=appearance["offset_end"],
            confidence=entity["confidence"],
            context_before=entity.get("context_before", ""),
            context_after=entity.get("context_after", ""),
            sentence=entity.get("sentence", "")
        )

# MERGE (e)-[m:MENTIONED_IN]->(c)
# SET m.offset_start, m.offset_end, m.confidence, m.context_*, m.sentence
```

5. **Create Semantic Relationships**:
```python
for rel in relationships:
    await self.neo4j_client.create_relationship(
        subject_normalized=rel["subject_normalized"],
        predicate=rel["predicate"],
        object_normalized=rel["object_normalized"],
        confidence=rel["confidence"],
        context=rel["context"]
    )

# MATCH (s:Entity {normalized: $subject}), (o:Entity {normalized: $object})
# MERGE (s)-[r:USES]->(o)  # Dynamic relationship type
# ON CREATE SET r.confidence, r.context, r.occurrence_count = 1
# ON MATCH SET r.confidence = rolling average, r.occurrence_count += 1
```

**Transaction Scope**: Each database operation is a separate transaction (auto-commit)

**Error Handling**: If any operation fails, exception propagates to caller (no partial commits)

### Response Formatting

**Method**: `_format_entities_for_response(entities) -> List[Dict]`

Formats entities from internal representation to API response model:
```python
{
  "text": entity["text"],
  "normalized": entity["normalized"],
  "type_primary": entity["type_primary"],
  "type_sub1": entity.get("type_sub1"),
  "type_sub2": entity.get("type_sub2"),
  "type_sub3": entity.get("type_sub3"),
  "type_full": entity["type_full"],
  "confidence": entity["confidence"],
  "context_before": entity.get("context_before", ""),
  "context_after": entity.get("context_after", ""),
  "sentence": entity.get("sentence", ""),
  "chunk_appearances": entity.get("chunk_appearances", []),
  "spans_multiple_chunks": entity.get("spans_multiple_chunks", False)
}
```

**Method**: `_format_relationships_for_response(relationships) -> List[Dict]`

Formats relationships from internal representation to API response model:
```python
{
  "subject_text": rel["subject_text"],
  "subject_normalized": rel["subject_normalized"],
  "predicate": rel["predicate"],
  "object_text": rel["object_text"],
  "object_normalized": rel["object_normalized"],
  "confidence": rel["confidence"],
  "context": rel["context"],
  "spans_chunks": rel.get("spans_chunks", False),
  "chunk_rowids": rel.get("chunk_rowids", [])
}
```

## Chunk Mapper

### Purpose

Map entities and relationships extracted from full documents to specific chunk boundaries, enabling precise retrieval while maintaining extraction context.

### Architecture

**Class**: `ChunkMapper`

**Data Classes**:
- `ChunkBoundary`: Chunk metadata (vector_rowid, chunk_index, char_start, char_end, text)
- `EntityChunkMapping`: Entity with chunk appearances (unused in current impl)
- `RelationshipChunkMapping`: Relationship with chunk associations (unused in current impl)

**Singleton Pattern**: Global instance via `get_chunk_mapper()`

### Entity-to-Chunk Mapping

**Method**: `map_entities_to_chunks(entities, chunks) -> List[Dict]`

**Purpose**: Calculate which chunks contain each entity mention

**Input**:
- `entities`: Entities with document positions (start, end)
- `chunks`: Chunk boundaries (vector_rowid, chunk_index, char_start, char_end)

**Output**: Entities enriched with chunk_appearances

**Algorithm**:

1. **Parse Chunk Boundaries**:
```python
chunk_boundaries = [
    ChunkBoundary(
        vector_rowid=chunk["vector_rowid"],
        chunk_index=chunk["chunk_index"],
        char_start=chunk["char_start"],
        char_end=chunk["char_end"],
        text=chunk.get("text", "")
    )
    for chunk in chunks
]
```

2. **For Each Entity**:
```python
for entity in entities:
    # Get entity occurrences (supports multiple mentions)
    occurrences = entity.get("occurrences", [])

    # If no occurrences, create from entity position
    if not occurrences:
        occurrences = [{
            "start": entity.get("start_pos", 0),
            "end": entity.get("end_pos", len(entity["text"])),
            "context": entity.get("sentence", "")
        }]
```

3. **For Each Occurrence**:
```python
for occurrence in occurrences:
    occ_start = occurrence["start"]
    occ_end = occurrence["end"]

    # Find overlapping chunks
    for chunk in chunk_boundaries:
        overlap = calculate_overlap(
            occ_start, occ_end,
            chunk.char_start, chunk.char_end
        )

        if overlap >= overlap_threshold (10 chars):
            # Calculate offset within chunk
            offset_start = max(0, occ_start - chunk.char_start)
            offset_end = min(len(chunk.text), occ_end - chunk.char_start)

            chunk_appearances.append({
                "vector_rowid": chunk.vector_rowid,
                "chunk_index": chunk.chunk_index,
                "offset_start": offset_start,
                "offset_end": offset_end
            })
```

4. **Enrich Entity**:
```python
enriched_entity = entity.copy()
enriched_entity["chunk_appearances"] = chunk_appearances
enriched_entity["spans_multiple_chunks"] = len(chunk_appearances) > 1
enriched_entity["num_chunks"] = len(chunk_appearances)
```

**Overlap Calculation**:
```python
def _calculate_overlap(start1, end1, start2, end2) -> int:
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0, overlap_end - overlap_start)
```

**Performance**: O(E * C * O) where E = entities, C = chunks, O = occurrences per entity
- Typical: 42 entities * 18 chunks * 1 occurrence = ~756 overlap checks
- Execution time: <100ms

### Relationship-to-Chunk Mapping

**Method**: `map_relationships_to_chunks(relationships, entity_chunk_map, chunks) -> List[Dict]`

**Purpose**: Determine which chunks are involved in each relationship

**Input**:
- `relationships`: Subject-predicate-object triples
- `entity_chunk_map`: Entities with chunk_appearances
- `chunks`: Chunk boundaries

**Output**: Relationships enriched with chunk information

**Algorithm**:

1. **Build Entity Lookup**:
```python
entity_lookup = {
    entity["normalized"]: entity
    for entity in entity_chunk_map
}
```

2. **For Each Relationship**:
```python
subject_entity = entity_lookup.get(rel["subject_normalized"])
object_entity = entity_lookup.get(rel["object_normalized"])

if not subject_entity or not object_entity:
    # Skip relationship if entities not found
    continue
```

3. **Find Involved Chunks**:
```python
subject_chunks = {
    app["vector_rowid"]
    for app in subject_entity["chunk_appearances"]
}

object_chunks = {
    app["vector_rowid"]
    for app in object_entity["chunk_appearances"]
}

all_chunks = subject_chunks | object_chunks
chunk_rowids = sorted(list(all_chunks))
```

4. **Check if Cross-Chunk**:
```python
spans_chunks = not bool(subject_chunks & object_chunks)
# True if entities have no shared chunks
```

5. **Find Primary Chunk**:
```python
primary_chunk_rowid = _find_primary_chunk(
    subject_entity,
    object_entity,
    chunks
)

# Priority:
# 1. Chunk where both entities appear
# 2. Chunk closest to both entities
# 3. First chunk containing subject
```

6. **Enrich Relationship**:
```python
enriched_rel = rel.copy()
enriched_rel["spans_chunks"] = spans_chunks
enriched_rel["chunk_rowids"] = chunk_rowids
enriched_rel["primary_chunk_rowid"] = primary_chunk_rowid
enriched_rel["num_chunks_involved"] = len(chunk_rowids)
```

**Primary Chunk Selection**:
```python
def _find_primary_chunk(subject_entity, object_entity, chunks):
    subject_chunks = {app["vector_rowid"] for app in subject_entity["chunk_appearances"]}
    object_chunks = {app["vector_rowid"] for app in object_entity["chunk_appearances"]}

    # Check for shared chunks
    shared = subject_chunks & object_chunks
    if shared:
        return min(shared)  # Return first shared chunk

    # Find closest chunks by index
    min_distance = float('inf')
    primary_rowid = None

    for subj_rowid in subject_chunks:
        for obj_rowid in object_chunks:
            distance = abs(get_chunk_index(subj_rowid) - get_chunk_index(obj_rowid))
            if distance < min_distance:
                min_distance = distance
                primary_rowid = min(subj_rowid, obj_rowid)

    return primary_rowid
```

### Mapping Summary

**Method**: `generate_mapping_summary(entities, relationships, chunks) -> Dict`

**Purpose**: Generate statistics about chunk mapping results

**Output**:
```python
{
  "total_chunks": 18,
  "chunks_with_entities": 15,
  "total_entity_appearances": 67,
  "unique_entities": 42,
  "multi_chunk_entities": 8,
  "avg_entities_per_chunk": 4.47,
  "total_relationships": 18,
  "cross_chunk_relationships": 3,
  "chunk_entity_distribution": {
    45003: 12,  # Chunk with most entities
    45001: 10,
    45007: 8,
    ...
  }
}
```

**Calculation**:
```python
# Count entity appearances
total_entity_appearances = sum(
    len(e["chunk_appearances"])
    for e in entities
)

# Count multi-chunk entities
multi_chunk_entities = sum(
    1 for e in entities
    if e.get("spans_multiple_chunks", False)
)

# Count cross-chunk relationships
cross_chunk_relationships = sum(
    1 for r in relationships
    if r.get("spans_chunks", False)
)

# Build chunk distribution
chunk_entity_count = {}
for entity in entities:
    for app in entity["chunk_appearances"]:
        rowid = app["vector_rowid"]
        chunk_entity_count[rowid] = chunk_entity_count.get(rowid, 0) + 1

# Sort and take top 10
top_10_chunks = dict(sorted(
    chunk_entity_count.items(),
    key=lambda x: x[1],
    reverse=True
)[:10])
```

## Error Handling

### Initialization Errors

**Neo4j Connection Failure**:
```python
try:
    connected = await self.neo4j_client.connect()
    if not connected:
        logger.error("Failed to connect to Neo4j")
        return False
except Exception as e:
    logger.error(f"Failed to initialize KG Processor: {e}")
    return False
```

**Result**: Service fails to start, returns 503 on all requests

### Processing Errors

**Entity Extraction Failure**:
- Exception logged with full traceback
- Propagates to caller (returns 500 to client)

**Relationship Extraction Failure**:
- vLLM unavailable: Returns empty relationship list (graceful degradation)
- Other errors: Exception propagates

**Chunk Mapping Failure**:
- Exception logged
- Propagates to caller

**Storage Failure**:
- Neo4j error logged
- Exception propagates
- No partial commits (each operation is atomic)

### Recovery Mechanisms

**vLLM Reconnection**:
- vLLM client resets state on failure
- Next request triggers model rediscovery
- Automatic retry with exponential backoff

**Neo4j Reconnection**:
- Driver handles reconnection automatically
- Transient errors retried by driver
- Permanent failures propagate

## Performance Optimization

### Concurrency

**Async Operations**:
- All I/O-bound operations use async/await
- Neo4j operations execute concurrently where possible
- vLLM requests are async (but sequential per document)

**Concurrent Document Processing**:
```python
# Process multiple documents concurrently
tasks = [
    processor.process_document(...)
    for doc in documents
]
results = await asyncio.gather(*tasks)
```

**Concurrency Limits**:
- MAX_CONCURRENT_REQUESTS (default 8)
- Neo4j connection pool (50 connections)
- vLLM client pool (10 connections)

### Memory Management

**Entity Storage**:
- Entities stored in Python lists (not persistent)
- Typical memory: ~42 entities * 1KB = ~42KB per document
- Cleared after processing

**Chunk Mapping**:
- Temporary data structures during mapping
- Released after storage

**Global State**:
- Processing stats stored in-memory (small footprint)
- No persistent caching

### Bottleneck Analysis

**Stage Timing**:
1. Entity Extraction: 2-3 seconds (GLiNER inference)
2. Relationship Extraction: 5-10 seconds (vLLM latency)
3. Chunk Mapping: <100ms (computational)
4. Neo4j Storage: 1-2 seconds (I/O)

**Total**: 10-15 seconds per document

**Limiting Factor**: vLLM inference latency

**Optimization Strategies**:
- Parallel document processing (multiple concurrent requests)
- Faster vLLM model (smaller/quantized)
- Caching relationship extractions (requires careful invalidation)

---

[Next: Storage Module Documentation](storage.md)
