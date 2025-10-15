# Knowledge Graph System - Retrieval & Query Plan (Part 2)

**Document Version:** 1.0
**Created:** 2025-10-15
**Status:** Planning Phase

---

## Table of Contents - Part 2

1. [Query Architecture Overview](#1-query-architecture-overview)
2. [Graph Exploration System](#2-graph-exploration-system)
3. [Entity & Relationship Traversal](#3-entity--relationship-traversal)
4. [Configurable Exploration Depth](#4-configurable-exploration-depth)
5. [Chunk Retrieval from RAG](#5-chunk-retrieval-from-rag)
6. [Result Combination & Ranking](#6-result-combination--ranking)
7. [Query Response Format](#7-query-response-format)
8. [API Integration Flow](#8-api-integration-flow)
9. [Performance Optimization](#9-performance-optimization)
10. [Configuration & Deployment](#10-configuration--deployment)

---

## 1. Query Architecture Overview

### 1.1 Core Principle

**Graph-First Query Pipeline:** Query Neo4j BEFORE vector search to discover related entities, then use graph intelligence to enhance semantic search.

### 1.2 Query Flow Diagram

```
User Query: "FastAPI best practices"
    │
    ├─ Step 1: Parse Query
    │  └─ Extract entities: ["FastAPI"]
    │  └─ Extract intent: "best practices" (informational)
    │
    ├─ Step 2: Neo4j Graph Exploration (FIRST!)
    │  │
    │  ├─ 2a. Find Matching Entities
    │  │   Query: MATCH (e:Entity) WHERE e.text =~ '(?i).*fastapi.*'
    │  │   Result: Entity(FastAPI, Framework::Backend::Python)
    │  │
    │  ├─ 2b. Traverse Relationships (depth=2)
    │  │   Query: MATCH (e)-[r:RELATED_TO*1..2]-(related)
    │  │   Results:
    │  │     - 1-hop: Pydantic (uses), Uvicorn (uses), Starlette (uses)
    │  │     - 2-hop: Python (via Pydantic), asyncio (via Uvicorn)
    │  │
    │  ├─ 2c. Find Co-occurring Entities
    │  │   Query: MATCH (e)-[:CO_OCCURS_WITH]-(cooccur)
    │  │   Results: PostgreSQL, Docker, pytest (frequently appear together)
    │  │
    │  └─ 2d. Get Chunks Containing These Entities
    │      Query: MATCH (related)<-[:CONTAINS]-(chunk:Chunk)
    │      Results: chunk_rowids [45001, 45003, 45123, 45201, ...]
    │
    ├─ Step 3: Expand Query with Graph Context
    │  │
    │  ├─ Original: "FastAPI best practices"
    │  └─ Expanded: "FastAPI Pydantic Uvicorn asyncio pytest best practices"
    │
    ├─ Step 4: Vector Search (SQLite + sqlite-vec)
    │  │
    │  ├─ Embed expanded query → 384-dim vector
    │  ├─ Query content_vectors for similarity
    │  └─ Results: Top 20 chunks by cosine similarity
    │
    ├─ Step 5: Combine & Rank Results
    │  │
    │  ├─ Graph chunks: Found via entity relationships
    │  ├─ Vector chunks: Found via semantic similarity
    │  │
    │  └─ Scoring: combined_score = (vector_sim * 0.6) + (graph_relevance * 0.4)
    │      where graph_relevance = (entity_match_count + relationship_count) / max_possible
    │
    ├─ Step 6: Fetch Full Content from SQLite
    │  │
    │  ├─ JOIN: content_vectors → content_chunks → crawled_content
    │  ├─ LEFT JOIN: chunk_entities (get entities in chunk)
    │  └─ LEFT JOIN: chunk_relationships (get relationships)
    │
    ├─ Step 7: Enrich with Graph Context
    │  │
    │  └─ For each result chunk:
    │      - Add entities found in this chunk
    │      - Add relationships involving these entities
    │      - Add related documents (via shared entities)
    │      - Add graph traversal path (how we found this chunk)
    │
    └─ Step 8: Return to User
       │
       └─ JSON response with:
           - Ranked results
           - Entity/relationship context
           - Graph exploration metadata
           - Suggested related queries
```

### 1.3 Query Pipeline Components

```python
# mcpragcrawl4ai/core/data/graph_search.py

class GraphEnhancedRAG:
    """
    Graph-enhanced semantic search combining Neo4j + vector search
    """

    def __init__(self, rag_db, neo4j_client, config):
        self.rag_db = rag_db
        self.neo4j = neo4j_client
        self.config = config

    async def search(
        self,
        query: str,
        limit: int = 10,
        tags: Optional[str] = None,
        exploration_depth: int = 2,
        graph_weight: float = 0.4,
        enable_graph: bool = True
    ) -> List[Dict]:
        """
        Main search pipeline with graph enhancement

        Args:
            query: User search query
            limit: Maximum results to return
            tags: Optional tag filter
            exploration_depth: How many hops in Neo4j (0-3)
            graph_weight: Weight for graph relevance (0.0-1.0)
            enable_graph: Enable/disable graph exploration

        Returns:
            List of results with graph context
        """

        # Step 1: Parse query
        query_entities = self._parse_query_entities(query)

        # Step 2: Neo4j exploration (if enabled)
        if enable_graph and exploration_depth > 0:
            graph_results = await self.neo4j.explore_query(
                query=query,
                entities=query_entities,
                depth=exploration_depth,
                limit=100
            )
            expanded_query = self._expand_query(query, graph_results)
        else:
            graph_results = []
            expanded_query = query

        # Step 3: Vector search
        vector_results = await self.rag_db.vector_search(
            query=expanded_query,
            limit=limit * 2,  # Get more for re-ranking
            tags=tags
        )

        # Step 4: Combine and rank
        combined = self._combine_results(
            vector_results=vector_results,
            graph_results=graph_results,
            graph_weight=graph_weight
        )

        # Step 5: Fetch content and enrich
        enriched = await self._enrich_results(combined[:limit])

        return enriched
```

---

## 2. Graph Exploration System

### 2.1 Entity Matching

**Goal:** Find entities in Neo4j that match the user's query.

#### Query 1: Exact Entity Match

```cypher
// Find exact entity matches (case-insensitive)
MATCH (e:Entity)
WHERE toLower(e.text) = toLower($query_term)
RETURN e.text, e.type_primary, e.type_sub1, e.type_sub2,
       elementId(e) as entity_id,
       e.total_mentions, e.document_count
ORDER BY e.total_mentions DESC
LIMIT 10
```

**Example:**
```
Input: "fastapi"
Output:
  - text: "FastAPI"
    type: Framework::Backend::Python
    mentions: 87
    documents: 12
```

---

#### Query 2: Fuzzy Entity Match (Text Search)

```cypher
// Find entities containing query term
MATCH (e:Entity)
WHERE toLower(e.text) CONTAINS toLower($query_term)
   OR toLower(e.normalized) CONTAINS toLower($query_term)
RETURN e.text, e.type_primary, e.type_sub1, e.type_sub2,
       elementId(e) as entity_id,
       e.total_mentions, e.document_count
ORDER BY e.total_mentions DESC
LIMIT 20
```

**Example:**
```
Input: "python web"
Output:
  - FastAPI (Framework::Backend::Python) - 87 mentions
  - Django (Framework::Backend::Python) - 65 mentions
  - Flask (Framework::Backend::Python) - 54 mentions
  - Python (Language::Interpreted::DynamicTyped) - 203 mentions
```

---

#### Query 3: Type Hierarchy Match

```cypher
// Find entities by type hierarchy
MATCH (e:Entity)
WHERE e.type_primary = $type_primary
  AND ($type_sub1 IS NULL OR e.type_sub1 = $type_sub1)
  AND ($type_sub2 IS NULL OR e.type_sub2 = $type_sub2)
RETURN e.text, e.type_primary, e.type_sub1, e.type_sub2,
       elementId(e) as entity_id
ORDER BY e.total_mentions DESC
LIMIT 50
```

**Example:**
```
Input: type_primary="Framework", type_sub1="Backend", type_sub2="Python"
Output:
  - FastAPI (Framework::Backend::Python)
  - Django (Framework::Backend::Python)
  - Flask (Framework::Backend::Python)
  - Pyramid (Framework::Backend::Python)
```

---

### 2.2 Python Implementation: Entity Matching

```python
# kg-service/storage/neo4j_queries.py (shared with mcpragcrawl4ai)

class Neo4jQueryEngine:
    """Neo4j query engine for graph exploration"""

    def __init__(self, driver):
        self.driver = driver

    def find_entities_by_text(self, query_term: str, limit: int = 20) -> List[Dict]:
        """Find entities matching query text"""

        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.text) CONTAINS toLower($query_term)
                   OR toLower(e.normalized) CONTAINS toLower($query_term)
                RETURN e.text as text,
                       e.type_primary as type_primary,
                       e.type_sub1 as type_sub1,
                       e.type_sub2 as type_sub2,
                       elementId(e) as entity_id,
                       e.total_mentions as mentions,
                       e.document_count as documents
                ORDER BY e.total_mentions DESC
                LIMIT $limit
            """, {"query_term": query_term, "limit": limit})

            return [dict(record) for record in result]

    def find_entities_by_type(
        self,
        type_primary: str,
        type_sub1: Optional[str] = None,
        type_sub2: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Find entities by type hierarchy"""

        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE e.type_primary = $type_primary
                  AND ($type_sub1 IS NULL OR e.type_sub1 = $type_sub1)
                  AND ($type_sub2 IS NULL OR e.type_sub2 = $type_sub2)
                RETURN e.text as text,
                       e.type_primary, e.type_sub1, e.type_sub2,
                       elementId(e) as entity_id,
                       e.total_mentions as mentions
                ORDER BY e.total_mentions DESC
                LIMIT $limit
            """, {
                "type_primary": type_primary,
                "type_sub1": type_sub1,
                "type_sub2": type_sub2,
                "limit": limit
            })

            return [dict(record) for record in result]
```

---

## 3. Entity & Relationship Traversal

### 3.1 Direct Relationship Queries

#### Query 4: 1-Hop Relationships (Direct Connections)

```cypher
// Find entities directly related to query entity
MATCH (e:Entity {text: $entity_text})-[r:RELATED_TO]-(related:Entity)
WHERE r.confidence >= $min_confidence
RETURN related.text as entity,
       related.type_primary as type,
       related.type_sub1 as subtype,
       type(r) as relationship_direction,
       r.predicate as relationship_type,
       r.confidence as confidence,
       r.context as context,
       r.document_id as document_id,
       elementId(related) as entity_id
ORDER BY r.confidence DESC
LIMIT 50
```

**Example:**
```
Input: entity_text="FastAPI", min_confidence=0.5
Output:
  - Pydantic (Framework::Data::Python) - "uses" - 0.88 - "FastAPI uses Pydantic for..."
  - Uvicorn (Tool::Server::ASGI) - "uses" - 0.85 - "FastAPI runs on Uvicorn"
  - Django (Framework::Backend::Python) - "competes_with" - 0.72 - "FastAPI vs Django"
```

---

#### Query 5: Multi-Hop Traversal (2-3 Hops)

```cypher
// Find entities within N hops of query entity
MATCH (e:Entity {text: $entity_text})-[r:RELATED_TO*1..$depth]-(related:Entity)
WHERE ALL(rel in r WHERE rel.confidence >= $min_confidence)
WITH related, r,
     length(r) as hop_distance,
     [rel in r | rel.predicate] as relationship_path,
     reduce(conf = 1.0, rel in r | conf * rel.confidence) as path_confidence
WHERE hop_distance <= $depth
  AND path_confidence >= $min_path_confidence
RETURN DISTINCT
       related.text as entity,
       related.type_primary as type,
       related.type_sub1 as subtype,
       hop_distance,
       relationship_path,
       path_confidence,
       elementId(related) as entity_id
ORDER BY hop_distance ASC, path_confidence DESC
LIMIT 100
```

**Parameters:**
- `$depth`: 1, 2, or 3 (configurable exploration depth)
- `$min_confidence`: 0.5 (filter low-confidence relationships)
- `$min_path_confidence`: 0.3 (filter weak multi-hop paths)

**Example:**
```
Input: entity_text="FastAPI", depth=2, min_confidence=0.5
Output:
  1-hop:
    - Pydantic (uses) - confidence: 0.88
    - Uvicorn (uses) - confidence: 0.85

  2-hop:
    - Python (via Pydantic → uses) - path_conf: 0.88 * 0.90 = 0.79
    - asyncio (via Uvicorn → uses) - path_conf: 0.85 * 0.82 = 0.70
    - typing (via Pydantic → uses) - path_conf: 0.88 * 0.75 = 0.66
```

---

#### Query 6: Relationship Type Filtering

```cypher
// Find relationships of specific types
MATCH (e:Entity {text: $entity_text})-[r:RELATED_TO]-(related:Entity)
WHERE r.predicate IN $predicate_list
  AND r.confidence >= $min_confidence
RETURN related.text as entity,
       r.predicate as relationship,
       r.confidence as confidence,
       r.context as context
ORDER BY r.confidence DESC
```

**Example:**
```
Input:
  entity_text="FastAPI"
  predicate_list=["uses", "depends_on", "built_with"]

Output:
  - Pydantic - "uses" - 0.88
  - Uvicorn - "uses" - 0.85
  - Starlette - "built_with" - 0.82
```

---

### 3.2 Co-occurrence Queries

#### Query 7: Find Co-occurring Entities

```cypher
// Find entities that frequently appear together
MATCH (e:Entity {text: $entity_text})-[co:CO_OCCURS_WITH]-(related:Entity)
WHERE co.count >= $min_count
  AND co.document_count >= $min_docs
RETURN related.text as entity,
       related.type_primary as type,
       co.count as occurrences,
       co.document_count as documents,
       co.avg_proximity as avg_distance,
       elementId(related) as entity_id
ORDER BY co.count DESC
LIMIT 50
```

**Example:**
```
Input: entity_text="FastAPI", min_count=5, min_docs=2
Output:
  - Pydantic - 23 occurrences across 8 documents (avg 150 chars apart)
  - PostgreSQL - 15 occurrences across 6 documents (avg 300 chars apart)
  - Docker - 12 occurrences across 5 documents (avg 500 chars apart)
```

---

### 3.3 Python Implementation: Relationship Traversal

```python
# kg-service/storage/neo4j_queries.py

class Neo4jQueryEngine:
    """... continued from above ..."""

    def traverse_relationships(
        self,
        entity_text: str,
        depth: int = 2,
        min_confidence: float = 0.5,
        predicate_filter: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Traverse entity relationships up to N hops

        Args:
            entity_text: Starting entity
            depth: Maximum hops (1-3)
            min_confidence: Minimum relationship confidence
            predicate_filter: Only include specific relationship types
            limit: Maximum results

        Returns:
            List of related entities with paths
        """

        predicate_where = ""
        if predicate_filter:
            predicate_where = "AND ALL(rel in r WHERE rel.predicate IN $predicates)"

        query = f"""
            MATCH (e:Entity {{text: $entity_text}})-[r:RELATED_TO*1..$depth]-(related:Entity)
            WHERE ALL(rel in r WHERE rel.confidence >= $min_confidence)
              {predicate_where}
            WITH related, r,
                 length(r) as hop_distance,
                 [rel in r | rel.predicate] as relationship_path,
                 reduce(conf = 1.0, rel in r | conf * rel.confidence) as path_confidence
            WHERE path_confidence >= 0.3
            RETURN DISTINCT
                   related.text as entity,
                   related.type_primary as type,
                   related.type_sub1 as subtype,
                   hop_distance,
                   relationship_path,
                   path_confidence,
                   elementId(related) as entity_id
            ORDER BY hop_distance ASC, path_confidence DESC
            LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(query, {
                "entity_text": entity_text,
                "depth": depth,
                "min_confidence": min_confidence,
                "predicates": predicate_filter,
                "limit": limit
            })

            return [dict(record) for record in result]

    def find_cooccurring_entities(
        self,
        entity_text: str,
        min_count: int = 3,
        min_docs: int = 2,
        limit: int = 50
    ) -> List[Dict]:
        """Find entities that frequently appear with query entity"""

        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {text: $entity_text})-[co:CO_OCCURS_WITH]-(related:Entity)
                WHERE co.count >= $min_count
                  AND co.document_count >= $min_docs
                RETURN related.text as entity,
                       related.type_primary as type,
                       related.type_sub1 as subtype,
                       co.count as occurrences,
                       co.document_count as documents,
                       co.avg_proximity as avg_distance,
                       co.documents as document_ids,
                       elementId(related) as entity_id
                ORDER BY co.count DESC
                LIMIT $limit
            """, {
                "entity_text": entity_text,
                "min_count": min_count,
                "min_docs": min_docs,
                "limit": limit
            })

            return [dict(record) for record in result]
```

---

## 4. Configurable Exploration Depth

### 4.1 Depth Parameter Design

**Purpose:** Allow users/system to control how "wide" the graph search goes.

**Depth Levels:**

| Depth | Description | Use Case | Performance |
|-------|-------------|----------|-------------|
| 0 | **No graph** - pure vector search | Speed-critical, simple queries | Fastest |
| 1 | **Direct relationships only** | Default for most queries | Fast |
| 2 | **2-hop traversal** | Exploratory queries, find hidden connections | Medium |
| 3 | **3-hop traversal** | Research mode, comprehensive discovery | Slower |

### 4.2 Depth Configuration

#### Docker Compose Environment Variable

```yaml
# docker-compose.yml for kg-service

services:
  kg-service:
    image: kg-service:latest
    environment:
      - KG_DEFAULT_EXPLORATION_DEPTH=2
      - KG_MAX_EXPLORATION_DEPTH=3
      - KG_MIN_RELATIONSHIP_CONFIDENCE=0.5
      - KG_GRAPH_QUERY_TIMEOUT=5000  # milliseconds
```

#### mcpragcrawl4ai Configuration

```python
# mcpragcrawl4ai/config.py (add these)

class Settings(BaseSettings):
    # ... existing settings ...

    # Knowledge Graph Query Settings
    KG_ENABLED: bool = Field(default=True, env="KG_ENABLED")
    KG_DEFAULT_DEPTH: int = Field(default=2, ge=0, le=3, env="KG_DEFAULT_DEPTH")
    KG_MAX_DEPTH: int = Field(default=3, env="KG_MAX_DEPTH")
    KG_MIN_CONFIDENCE: float = Field(default=0.5, env="KG_MIN_CONFIDENCE")
    KG_GRAPH_WEIGHT: float = Field(default=0.4, ge=0.0, le=1.0, env="KG_GRAPH_WEIGHT")
    KG_TIMEOUT_MS: int = Field(default=5000, env="KG_TIMEOUT_MS")

    # Neo4j Connection (for queries)
    NEO4J_URI: str = Field(default="bolt://neo4j-kg:7687", env="NEO4J_URI")
    NEO4J_USER: str = Field(default="neo4j", env="NEO4J_USER")
    NEO4J_PASSWORD: str = Field(default="knowledge_graph_2024", env="NEO4J_PASSWORD")
```

#### Per-Query Override

```python
# MCP tool with depth parameter

def search_memory(
    query: str,
    limit: int = 10,
    tags: str = "",
    graph_depth: Optional[int] = None,  # Override default
    enable_graph: bool = True
) -> List[Dict]:
    """
    Search memory with configurable graph exploration

    Args:
        query: Search query
        limit: Max results
        tags: Tag filter
        graph_depth: Override exploration depth (0-3)
            - 0: No graph (pure vector)
            - 1: Direct relationships
            - 2: 2-hop traversal (default)
            - 3: 3-hop traversal (comprehensive)
        enable_graph: Enable/disable graph enhancement

    Returns:
        List of results with graph context
    """

    # Use provided depth or fall back to config default
    depth = graph_depth if graph_depth is not None else settings.KG_DEFAULT_DEPTH

    # Clamp to max
    depth = min(depth, settings.KG_MAX_DEPTH)

    return GLOBAL_DB.search_with_graph(
        query=query,
        limit=limit,
        tags=tags,
        exploration_depth=depth,
        enable_graph=enable_graph
    )
```

---

### 4.3 Depth Query Implementation

```python
# mcpragcrawl4ai/core/data/graph_search.py

class GraphEnhancedRAG:
    """... continued ..."""

    async def explore_graph(
        self,
        query_entities: List[str],
        depth: int = 2,
        min_confidence: float = 0.5
    ) -> Dict:
        """
        Explore Neo4j graph starting from query entities

        Returns:
            {
                "entities": [...],
                "chunks": [...],
                "relationships": [...],
                "exploration_metadata": {...}
            }
        """

        if depth == 0:
            return {"entities": [], "chunks": [], "relationships": []}

        all_entities = []
        all_chunks = set()
        all_relationships = []

        for entity_text in query_entities:
            # Find entity in Neo4j
            entity_matches = self.neo4j.find_entities_by_text(entity_text, limit=5)

            for entity in entity_matches:
                # Traverse relationships based on depth
                if depth >= 1:
                    related = self.neo4j.traverse_relationships(
                        entity_text=entity["text"],
                        depth=depth,
                        min_confidence=min_confidence,
                        limit=100
                    )
                    all_entities.extend(related)

                # Get co-occurring entities
                if depth >= 2:
                    cooccur = self.neo4j.find_cooccurring_entities(
                        entity_text=entity["text"],
                        min_count=3,
                        min_docs=2,
                        limit=50
                    )
                    all_entities.extend(cooccur)

        # Get chunks containing these entities
        entity_ids = [e["entity_id"] for e in all_entities]
        chunks = self.neo4j.get_chunks_for_entities(entity_ids, limit=200)

        all_chunks.update(c["vector_rowid"] for c in chunks)

        return {
            "entities": all_entities,
            "chunks": list(all_chunks),
            "relationships": all_relationships,
            "exploration_metadata": {
                "depth": depth,
                "entities_found": len(all_entities),
                "chunks_found": len(all_chunks),
                "query_entities": query_entities
            }
        }
```

---

## 5. Chunk Retrieval from RAG

### 5.1 Chunk Retrieval Query

#### Query 8: Get Chunks for Entities

```cypher
// Find chunks containing specific entities
MATCH (e:Entity)-[:CONTAINS]-(chunk:Chunk)
WHERE elementId(e) IN $entity_ids
RETURN DISTINCT
       chunk.vector_rowid as chunk_id,
       chunk.content_id as document_id,
       chunk.chunk_index as chunk_index,
       COLLECT(DISTINCT e.text) as entities_in_chunk,
       COUNT(DISTINCT e) as entity_count
ORDER BY entity_count DESC
LIMIT $limit
```

**Example:**
```
Input: entity_ids=[id1, id2, id3] (FastAPI, Pydantic, Uvicorn)
Output:
  - chunk_id: 45001 - entities: [FastAPI, Pydantic] - count: 2
  - chunk_id: 45003 - entities: [FastAPI, Uvicorn] - count: 2
  - chunk_id: 45123 - entities: [FastAPI] - count: 1
```

---

#### Query 9: Get Chunks with Relationship Context

```cypher
// Find chunks where specific relationships occur
MATCH (subj:Entity)-[r:RELATED_TO]->(obj:Entity)
WHERE elementId(subj) IN $entity_ids
   OR elementId(obj) IN $entity_ids
WITH subj, r, obj
UNWIND r.chunk_ids as chunk_id
MATCH (c:Chunk {vector_rowid: chunk_id})
RETURN DISTINCT
       c.vector_rowid as chunk_id,
       c.content_id as document_id,
       subj.text as subject,
       r.predicate as relationship,
       obj.text as object,
       r.confidence as confidence
ORDER BY r.confidence DESC
LIMIT $limit
```

---

### 5.2 SQLite Content Retrieval

```python
# mcpragcrawl4ai/core/data/storage.py

class RAGDatabase:
    """... existing class ..."""

    def fetch_chunks_by_rowids(
        self,
        chunk_rowids: List[int],
        include_entities: bool = True
    ) -> List[Dict]:
        """
        Fetch full chunk content from SQLite by vector_rowids

        Args:
            chunk_rowids: List of content_vectors rowids
            include_entities: Include entity information

        Returns:
            List of chunks with content and metadata
        """

        if not chunk_rowids:
            return []

        # Build placeholders for IN clause
        placeholders = ','.join('?' * len(chunk_rowids))

        # Base query
        query = f"""
            SELECT
                cc.rowid as chunk_rowid,
                cc.content_id,
                cc.chunk_index,
                cc.chunk_text,
                cc.char_start,
                cc.char_end,
                c.url,
                c.title,
                c.tags,
                c.kg_entity_count,
                c.kg_relationship_count
            FROM content_chunks cc
            JOIN crawled_content c ON cc.content_id = c.id
            WHERE cc.rowid IN ({placeholders})
            ORDER BY cc.content_id, cc.chunk_index
        """

        cursor = self.db.execute(query, chunk_rowids)
        chunks = []

        for row in cursor.fetchall():
            chunk_data = {
                "chunk_rowid": row[0],
                "content_id": row[1],
                "chunk_index": row[2],
                "chunk_text": row[3],
                "char_start": row[4],
                "char_end": row[5],
                "url": row[6],
                "title": row[7],
                "tags": row[8].split(",") if row[8] else [],
                "kg_entity_count": row[9],
                "kg_relationship_count": row[10]
            }

            # Optionally fetch entities in this chunk
            if include_entities:
                chunk_data["entities"] = self._get_chunk_entities(row[0])
                chunk_data["relationships"] = self._get_chunk_relationships(row[1])

            chunks.append(chunk_data)

        return chunks

    def _get_chunk_entities(self, chunk_rowid: int) -> List[Dict]:
        """Get entities in specific chunk"""

        cursor = self.db.execute("""
            SELECT
                entity_text,
                entity_type_primary,
                entity_type_sub1,
                entity_type_sub2,
                confidence,
                offset_start,
                offset_end,
                neo4j_node_id
            FROM chunk_entities
            WHERE chunk_rowid = ?
            ORDER BY offset_start
        """, (chunk_rowid,))

        return [
            {
                "text": row[0],
                "type_primary": row[1],
                "type_sub1": row[2],
                "type_sub2": row[3],
                "confidence": row[4],
                "offset_start": row[5],
                "offset_end": row[6],
                "neo4j_node_id": row[7]
            }
            for row in cursor.fetchall()
        ]

    def _get_chunk_relationships(self, content_id: int) -> List[Dict]:
        """Get relationships in document"""

        cursor = self.db.execute("""
            SELECT
                subject_entity,
                predicate,
                object_entity,
                confidence,
                context,
                spans_chunks,
                chunk_rowids
            FROM chunk_relationships
            WHERE content_id = ?
            ORDER BY confidence DESC
        """, (content_id,))

        return [
            {
                "subject": row[0],
                "predicate": row[1],
                "object": row[2],
                "confidence": row[3],
                "context": row[4],
                "spans_chunks": bool(row[5]),
                "chunk_rowids": json.loads(row[6]) if row[6] else []
            }
            for row in cursor.fetchall()
        ]
```

---

## 6. Result Combination & Ranking

### 6.1 Scoring Algorithm

**Combined Score Formula:**
```python
combined_score = (vector_similarity * vector_weight) + (graph_relevance * graph_weight)

where:
    vector_weight = 1.0 - graph_weight  # Typically 0.6
    graph_weight = configurable (default 0.4)

    vector_similarity = cosine similarity from sqlite-vec (0.0-1.0)

    graph_relevance = (entity_match_score + relationship_score) / 2

    entity_match_score = (matching_entities_count / total_query_entities) * entity_weight
    relationship_score = (relationship_count / max_relationships) * rel_weight
```

### 6.2 Ranking Implementation

```python
# mcpragcrawl4ai/core/data/graph_search.py

class GraphEnhancedRAG:
    """... continued ..."""

    def _combine_results(
        self,
        vector_results: List[Dict],
        graph_results: Dict,
        graph_weight: float = 0.4
    ) -> List[Dict]:
        """
        Combine and rank vector + graph results

        Args:
            vector_results: Results from vector search
            graph_results: Results from Neo4j exploration
            graph_weight: Weight for graph relevance (0.0-1.0)

        Returns:
            Combined and ranked results
        """

        vector_weight = 1.0 - graph_weight
        graph_chunk_ids = set(graph_results.get("chunks", []))
        graph_entity_map = {
            e["entity_id"]: e
            for e in graph_results.get("entities", [])
        }

        combined = []

        for vec_result in vector_results:
            chunk_rowid = vec_result["chunk_rowid"]
            vector_sim = vec_result["similarity"]

            # Calculate graph relevance
            is_in_graph = chunk_rowid in graph_chunk_ids

            if is_in_graph:
                # Count matching entities in this chunk
                chunk_entities = vec_result.get("entities", [])
                matching_entities = sum(
                    1 for e in chunk_entities
                    if e.get("neo4j_node_id") in graph_entity_map
                )

                # Count relationships
                chunk_rels = vec_result.get("relationships", [])
                relationship_count = len(chunk_rels)

                # Calculate graph relevance
                entity_score = min(1.0, matching_entities / max(1, len(graph_entity_map)))
                rel_score = min(1.0, relationship_count / 10)  # Normalize by max 10 rels
                graph_relevance = (entity_score * 0.7) + (rel_score * 0.3)
            else:
                graph_relevance = 0.0

            # Combined score
            combined_score = (vector_sim * vector_weight) + (graph_relevance * graph_weight)

            combined.append({
                **vec_result,
                "vector_similarity": vector_sim,
                "graph_relevance": graph_relevance,
                "combined_score": combined_score,
                "found_by_graph": is_in_graph
            })

        # Sort by combined score
        combined.sort(key=lambda x: x["combined_score"], reverse=True)

        return combined

    def _calculate_graph_relevance(
        self,
        chunk_entities: List[Dict],
        chunk_relationships: List[Dict],
        graph_entities: Dict,
        query_entities: List[str]
    ) -> float:
        """
        Calculate how relevant this chunk is based on graph exploration

        Factors:
        - How many query entities are in this chunk?
        - How many related entities (from graph traversal) are in this chunk?
        - How many relationships are in this chunk?
        - Entity type relevance
        """

        # Entity matching
        query_entity_matches = sum(
            1 for e in chunk_entities
            if e["text"].lower() in [q.lower() for q in query_entities]
        )

        graph_entity_matches = sum(
            1 for e in chunk_entities
            if e.get("neo4j_node_id") in graph_entities
        )

        # Relationship score
        relationship_score = min(1.0, len(chunk_relationships) / 5)

        # Weighted combination
        entity_match_score = min(1.0, (query_entity_matches * 2 + graph_entity_matches) / 10)

        return (entity_match_score * 0.7) + (relationship_score * 0.3)
```

---

### 6.3 Deduplication

```python
def _deduplicate_chunks(self, results: List[Dict]) -> List[Dict]:
    """
    Remove duplicate chunks (same chunk_rowid)
    Keep highest-scoring instance
    """

    seen = {}
    for result in results:
        chunk_id = result["chunk_rowid"]
        if chunk_id not in seen or result["combined_score"] > seen[chunk_id]["combined_score"]:
            seen[chunk_id] = result

    return list(seen.values())
```

---

## 7. Query Response Format

### 7.1 Response Structure

```python
{
  "success": true,
  "query": "FastAPI best practices",
  "results_count": 10,
  "total_found": 47,
  "exploration_depth": 2,
  "processing_time_ms": 342,

  "exploration_summary": {
    "graph_enabled": true,
    "entities_explored": 15,
    "relationships_traversed": 23,
    "chunks_from_graph": 38,
    "chunks_from_vector": 20,
    "chunks_combined": 45
  },

  "results": [
    {
      # Basic chunk info
      "chunk_rowid": 45001,
      "content_id": 123,
      "chunk_index": 0,
      "url": "https://docs.fastapi.com/tutorial",
      "title": "FastAPI Tutorial - Getting Started",

      # Content
      "chunk_text": "FastAPI is a modern, fast web framework...",
      "preview": "FastAPI is a modern, fast web framework for building APIs...",

      # Scoring
      "rank": 1,
      "combined_score": 0.92,
      "vector_similarity": 0.89,
      "graph_relevance": 0.95,
      "found_by_graph": true,

      # Graph context
      "entities": [
        {
          "text": "FastAPI",
          "type": "Framework::Backend::Python",
          "confidence": 0.95,
          "offset_start": 0,
          "offset_end": 7,
          "neo4j_node_id": "4:entity:789"
        },
        {
          "text": "Pydantic",
          "type": "Framework::Data::Python",
          "confidence": 0.92,
          "offset_start": 154,
          "offset_end": 162,
          "neo4j_node_id": "4:entity:790"
        }
      ],

      "relationships": [
        {
          "subject": "FastAPI",
          "predicate": "uses",
          "object": "Pydantic",
          "confidence": 0.88,
          "context": "FastAPI uses Pydantic for data validation",
          "spans_chunks": false
        }
      ],

      # Related content
      "related_documents": [
        {
          "url": "https://docs.pydantic.com",
          "title": "Pydantic Documentation",
          "shared_entities": ["Pydantic", "Python", "typing"],
          "relationship": "FastAPI uses Pydantic"
        }
      ],

      # Metadata
      "tags": ["python", "web", "api"],
      "kg_entity_count": 12,
      "kg_relationship_count": 5
    }
    // ... more results
  ],

  "suggested_queries": [
    "Pydantic data validation with FastAPI",
    "FastAPI deployment with Docker",
    "FastAPI vs Django performance"
  ],

  "related_entities": [
    {
      "text": "Pydantic",
      "type": "Framework::Data::Python",
      "relationship_to_query": "FastAPI uses Pydantic",
      "mentions": 23,
      "documents": 8
    },
    {
      "text": "Uvicorn",
      "type": "Tool::Server::ASGI",
      "relationship_to_query": "FastAPI runs on Uvicorn",
      "mentions": 15,
      "documents": 6
    }
  ]
}
```

---

## 8. API Integration Flow

### 8.1 MCP Tool Interface

```python
# mcpragcrawl4ai/core/rag_processor.py

async def search_memory_handler(arguments: dict) -> dict:
    """
    MCP tool: search_memory
    Enhanced with graph exploration
    """

    query = arguments.get("query", "")
    limit = arguments.get("limit", 10)
    tags = arguments.get("tags", "")
    graph_depth = arguments.get("graph_depth")  # Optional override
    enable_graph = arguments.get("enable_graph", True)

    try:
        # Use graph-enhanced search
        results = await graph_rag.search(
            query=query,
            limit=limit,
            tags=tags,
            exploration_depth=graph_depth,
            enable_graph=enable_graph
        )

        return {
            "success": True,
            "results": results,
            "metadata": {
                "query": query,
                "count": len(results),
                "graph_enabled": enable_graph,
                "exploration_depth": graph_depth or settings.KG_DEFAULT_DEPTH
            }
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "fallback_to_vector": True
        }
```

---

### 8.2 REST API Endpoint

```python
# mcpragcrawl4ai/api/api.py

@app.post("/api/v1/search")
async def search_endpoint(
    request: SearchRequest,
    session_info: Dict = Depends(verify_api_key)
):
    """
    Search endpoint with graph enhancement

    POST /api/v1/search
    {
      "query": "FastAPI best practices",
      "limit": 10,
      "tags": "python,api",
      "exploration_depth": 2,
      "enable_graph": true,
      "graph_weight": 0.4
    }
    """

    try:
        results = await graph_rag.search(
            query=request.query,
            limit=request.limit,
            tags=request.tags,
            exploration_depth=request.exploration_depth,
            enable_graph=request.enable_graph
        )

        return {
            "success": True,
            "data": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=100)
    tags: Optional[str] = None
    exploration_depth: Optional[int] = Field(default=None, ge=0, le=3)
    enable_graph: bool = Field(default=True)
    graph_weight: float = Field(default=0.4, ge=0.0, le=1.0)
```

---

## 9. Performance Optimization

### 9.1 Query Optimization

**Neo4j Indexes (from Part 1):**
```cypher
CREATE INDEX entity_text IF NOT EXISTS FOR (e:Entity) ON (e.text);
CREATE INDEX entity_type_hierarchy IF NOT EXISTS FOR (e:Entity) ON (e.type_primary, e.type_sub1, e.type_sub2);
CREATE INDEX relationship_predicate IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.predicate);
```

**SQLite Indexes:**
```sql
CREATE INDEX idx_chunk_entities_chunk ON chunk_entities(chunk_rowid);
CREATE INDEX idx_chunk_entities_entity ON chunk_entities(entity_text);
CREATE INDEX idx_chunk_entities_neo4j ON chunk_entities(neo4j_node_id);
CREATE INDEX idx_content_chunks_content_id ON content_chunks(content_id);
```

---

### 9.2 Caching Strategy

```python
# mcpragcrawl4ai/core/data/cache.py

from functools import lru_cache
import time

class QueryCache:
    """Cache for expensive graph queries"""

    def __init__(self, ttl: int = 3600):
        self.cache = {}
        self.ttl = ttl

    def get(self, key: str):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value):
        self.cache[key] = (value, time.time())

# Usage
query_cache = QueryCache(ttl=1800)  # 30 minutes

def explore_graph_cached(query, depth):
    cache_key = f"{query}:{depth}"
    cached = query_cache.get(cache_key)
    if cached:
        return cached

    result = explore_graph(query, depth)
    query_cache.set(cache_key, result)
    return result
```

---

### 9.3 Timeout Handling

```python
# mcpragcrawl4ai/core/data/graph_search.py

async def explore_graph_with_timeout(
    self,
    query: str,
    depth: int,
    timeout_ms: int = 5000
) -> Dict:
    """
    Graph exploration with timeout
    Falls back to empty results on timeout
    """

    try:
        result = await asyncio.wait_for(
            self.explore_graph(query, depth),
            timeout=timeout_ms / 1000.0
        )
        return result

    except asyncio.TimeoutError:
        logger.warning(f"Graph exploration timed out after {timeout_ms}ms")
        return {
            "entities": [],
            "chunks": [],
            "relationships": [],
            "timed_out": True
        }
```

---

## 10. Configuration & Deployment

### 10.1 Docker Compose Configuration

```yaml
# docker-compose.yml (updated for full KG system)

version: '3.8'

networks:
  crawler_default:
    external: true

services:
  # Neo4j Graph Database
  neo4j-kg:
    image: neo4j:5.25-community
    container_name: neo4j-kg
    restart: unless-stopped
    ports:
      - "7474:7474"  # HTTP Browser
      - "7687:7687"  # Bolt Protocol
    volumes:
      - ./neo4j/data:/data
      - ./neo4j/logs:/logs
      - ./neo4j/import:/var/lib/neo4j/import
      - ./neo4j/plugins:/plugins
    environment:
      - NEO4J_AUTH=neo4j/knowledge_graph_2024
      - NEO4J_server_memory_heap_initial__size=512m
      - NEO4J_server_memory_heap_max__size=2G
      - NEO4J_server_memory_pagecache_size=1G
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
    networks:
      - crawler_default
    healthcheck:
      test: ["CMD-SHELL", "wget --no-verbose --tries=1 --spider localhost:7474 || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s

  # KG Processing Service
  kg-service:
    build:
      context: ./kg-service
      dockerfile: Dockerfile
    container_name: kg-service
    restart: unless-stopped
    ports:
      - "8088:8088"  # REST API
    volumes:
      - ./kg-service/taxonomy:/app/taxonomy:ro
      - ./kg-service/logs:/app/logs
    environment:
      # Service config
      - API_HOST=0.0.0.0
      - API_PORT=8088
      - DEBUG=false
      - LOG_LEVEL=INFO

      # Neo4j config
      - NEO4J_URI=bolt://neo4j-kg:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=knowledge_graph_2024

      # vLLM config (external server on host)
      - VLLM_BASE_URL=http://host.docker.internal:8078
      - VLLM_TIMEOUT=120
      - VLLM_MAX_TOKENS=4096
      - VLLM_TEMPERATURE=0.1

      # GLiNER config
      - GLINER_MODEL=urchade/gliner_large-v2.1
      - GLINER_THRESHOLD=0.5
      - GLINER_BATCH_SIZE=8

      # Processing config
      - ENTITY_MIN_CONFIDENCE=0.5
      - RELATION_MIN_CONFIDENCE=0.6
    networks:
      - crawler_default
    depends_on:
      neo4j-kg:
        condition: service_healthy
    extra_hosts:
      - "host.docker.internal:host-gateway"  # Access host vLLM server
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8088/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # mcpragcrawl4ai RAG Service (updated)
  crawl4ai-rag-server:
    build:
      context: /home/robiloo/Documents/raid/mcpragcrawl4ai
      dockerfile: deployments/server/Dockerfile.api
    container_name: crawl4ai-rag-server
    restart: unless-stopped
    ports:
      - "8080:8080"  # REST API
      - "3000:3000"  # MCP Server
    volumes:
      - /home/robiloo/Documents/raid/mcpragcrawl4ai/data:/app/data
    environment:
      # Existing config
      - CRAWL4AI_URL=http://crawl4ai:11235
      - IS_SERVER=true
      - USE_MEMORY_DB=true

      # NEW: KG Integration config
      - KG_SERVICE_ENABLED=true
      - KG_SERVICE_URL=http://kg-service:8088
      - KG_SERVICE_TIMEOUT=300000  # 5 minutes

      # NEW: Neo4j Query config (for searches)
      - NEO4J_URI=bolt://neo4j-kg:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=knowledge_graph_2024

      # NEW: Graph query settings
      - KG_ENABLED=true
      - KG_DEFAULT_DEPTH=2
      - KG_MAX_DEPTH=3
      - KG_MIN_CONFIDENCE=0.5
      - KG_GRAPH_WEIGHT=0.4
      - KG_TIMEOUT_MS=5000
    networks:
      - crawler_default
    depends_on:
      - neo4j-kg
      - kg-service
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

### 10.2 Environment Variable Reference

#### kg-service

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | 0.0.0.0 | Service bind address |
| `API_PORT` | 8088 | Service port |
| `NEO4J_URI` | bolt://neo4j-kg:7687 | Neo4j connection |
| `VLLM_BASE_URL` | http://host.docker.internal:8078 | vLLM server |
| `GLINER_MODEL` | urchade/gliner_large-v2.1 | Entity extraction model |
| `GLINER_THRESHOLD` | 0.5 | Minimum entity confidence |
| `ENTITY_MIN_CONFIDENCE` | 0.5 | Filter low-confidence entities |
| `RELATION_MIN_CONFIDENCE` | 0.6 | Filter low-confidence relationships |

#### mcpragcrawl4ai (additions)

| Variable | Default | Description |
|----------|---------|-------------|
| `KG_SERVICE_ENABLED` | true | Enable KG processing |
| `KG_SERVICE_URL` | http://kg-service:8088 | kg-service API |
| `KG_SERVICE_TIMEOUT` | 300000 | KG processing timeout (ms) |
| `KG_ENABLED` | true | Enable graph-enhanced search |
| `KG_DEFAULT_DEPTH` | 2 | Default exploration depth |
| `KG_MAX_DEPTH` | 3 | Maximum allowed depth |
| `KG_MIN_CONFIDENCE` | 0.5 | Min relationship confidence |
| `KG_GRAPH_WEIGHT` | 0.4 | Weight for graph scoring |
| `KG_TIMEOUT_MS` | 5000 | Graph query timeout |

---

### 10.3 Startup Sequence

```bash
# 1. Ensure network exists
docker network create crawler_default

# 2. Start Neo4j first (required dependency)
docker compose up -d neo4j-kg

# Wait for Neo4j to be ready
docker compose logs -f neo4j-kg
# Look for: "Started."

# 3. Start kg-service
docker compose up -d kg-service

# 4. Start mcpragcrawl4ai (starts background KG worker)
docker compose up -d crawl4ai-rag-server

# 5. Verify all services
docker compose ps
docker compose logs -f
```

---

## Summary: Part 2 Complete

This document covers:
- ✅ Graph-first query architecture
- ✅ Entity matching and relationship traversal
- ✅ Configurable exploration depth (0-3 hops)
- ✅ Chunk retrieval from SQLite RAG
- ✅ Result combination and ranking algorithms
- ✅ Complete query response format
- ✅ API integration between services
- ✅ Performance optimization (caching, timeouts, indexes)
- ✅ Docker compose configuration with all environment variables

**Combined with Part 1 (KGPlan.md), you now have:**
- Complete data ingestion pipeline (crawl → KG → Neo4j)
- Complete query pipeline (Neo4j → graph → vector → combine → rank)
- Full Docker deployment configuration
- All database schemas (SQLite + Neo4j)
- Performance tuning and optimization
- Configurable parameters for tuning

**Ready for implementation!**

---

**Status:** Planning complete. Ready to begin building components.
