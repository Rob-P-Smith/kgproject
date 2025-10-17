"""
Search endpoints for KG-enhanced RAG retrieval

Provides entity search, chunk retrieval, and entity expansion
for use by mcpragcrawl4ai search pipeline.
"""

import logging
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from storage.neo4j_client import get_neo4j_client

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["Search"])


# ============================================================================
# Request/Response Models
# ============================================================================

class EntitySearchRequest(BaseModel):
    """Request to search for entities by text"""

    entity_terms: List[str] = Field(
        ...,
        description="Entity names/terms to search for",
        min_items=1
    )
    limit: int = Field(
        50,
        description="Max results per term",
        ge=1,
        le=500
    )
    min_mentions: int = Field(
        1,
        description="Minimum mention count",
        ge=1
    )

    class Config:
        schema_extra = {
            "example": {
                "entity_terms": ["neo4j", "python"],
                "limit": 50,
                "min_mentions": 1
            }
        }


class EntityMatch(BaseModel):
    """Matched entity from search"""

    entity_id: str = Field(..., description="Neo4j element ID")
    text: str = Field(..., description="Entity text")
    normalized: str = Field(..., description="Normalized text")
    type_primary: str = Field(..., description="Primary type")
    type_full: str = Field(..., description="Full hierarchical type")
    mention_count: int = Field(..., description="Total mentions")
    confidence: float = Field(..., description="Average confidence")


class EntitySearchResponse(BaseModel):
    """Response from entity search"""

    success: bool = True
    entities: List[EntityMatch] = Field(..., description="Matched entities")
    total_found: int = Field(..., description="Total entities found")


class ChunkSearchRequest(BaseModel):
    """Request to find chunks related to entities"""

    entity_ids: Optional[List[str]] = Field(
        None,
        description="Neo4j element IDs of entities"
    )
    entity_names: Optional[List[str]] = Field(
        None,
        description="Entity text names (alternative to IDs)"
    )
    limit: int = Field(
        100,
        description="Max chunks to return",
        ge=1,
        le=1000
    )
    include_document_info: bool = Field(
        True,
        description="Include document URL/title"
    )

    class Config:
        schema_extra = {
            "example": {
                "entity_names": ["Neo4j", "Python"],
                "limit": 100,
                "include_document_info": True
            }
        }


class ChunkMatch(BaseModel):
    """Chunk containing entity mentions"""

    chunk_id: str = Field(..., description="Neo4j Chunk node element ID")
    vector_rowid: int = Field(..., description="SQLite vector rowid")
    chunk_index: int = Field(..., description="Chunk index in document")
    entity_count: int = Field(..., description="Number of matched entities")
    matched_entities: List[str] = Field(..., description="Entity names in chunk")
    document_url: Optional[str] = Field(None, description="Document URL")
    document_title: Optional[str] = Field(None, description="Document title")


class ChunkSearchResponse(BaseModel):
    """Response from chunk search"""

    success: bool = True
    chunks: List[ChunkMatch] = Field(..., description="Matched chunks")
    total_found: int = Field(..., description="Total chunks found")


class EntityExpansionRequest(BaseModel):
    """Request to expand entities via relationships"""

    entity_names: List[str] = Field(
        ...,
        description="Entity names to expand from",
        min_items=1
    )
    max_expansions: int = Field(
        10,
        description="Max related entities to return",
        ge=1,
        le=100
    )
    min_confidence: float = Field(
        0.3,
        description="Minimum relationship confidence",
        ge=0.0,
        le=1.0
    )
    expansion_depth: int = Field(
        1,
        description="Relationship traversal depth",
        ge=1,
        le=3
    )

    class Config:
        schema_extra = {
            "example": {
                "entity_names": ["Neo4j"],
                "max_expansions": 10,
                "min_confidence": 0.3,
                "expansion_depth": 1
            }
        }


class RelatedEntity(BaseModel):
    """Entity discovered through expansion"""

    entity_id: str
    text: str
    normalized: str
    type_primary: str
    type_full: str
    mention_count: int
    relationship_type: Optional[str] = None
    relationship_confidence: Optional[float] = None
    path_distance: int = Field(..., description="Hops from original entity")


class EntityExpansionResponse(BaseModel):
    """Response from entity expansion"""

    success: bool = True
    original_entities: List[str] = Field(..., description="Input entities")
    expanded_entities: List[RelatedEntity] = Field(..., description="Related entities")
    total_discovered: int = Field(..., description="Total new entities")


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/search/entities", response_model=EntitySearchResponse)
async def search_entities(request: EntitySearchRequest):
    """
    Search for entities by text matching

    Searches Entity nodes where text contains any of the search terms.
    Used by GraphRetriever to find entities matching query terms.
    """
    try:
        client = get_neo4j_client()

        all_entities = []
        seen_ids = set()

        async with client.driver.session(database=client.database) as session:
            for term in request.entity_terms:
                # Query entities containing search term (case-insensitive)
                cypher = """
                MATCH (e:Entity)
                WHERE toLower(e.text) CONTAINS toLower($term)
                   OR toLower(e.normalized) CONTAINS toLower($term)
                WITH e
                WHERE e.mention_count >= $min_mentions
                RETURN
                    elementId(e) as entity_id,
                    e.text as text,
                    e.normalized as normalized,
                    e.type_primary as type_primary,
                    e.type_full as type_full,
                    e.mention_count as mention_count,
                    COALESCE(e.avg_confidence, 0.5) as confidence
                ORDER BY e.mention_count DESC
                LIMIT $limit
                """

                result = await session.run(
                    cypher,
                    term=term,
                    min_mentions=request.min_mentions,
                    limit=request.limit
                )

                records = await result.data()

                for record in records:
                    entity_id = record["entity_id"]
                    if entity_id not in seen_ids:
                        seen_ids.add(entity_id)
                        all_entities.append(EntityMatch(**record))

        logger.info(
            f"Entity search: {len(request.entity_terms)} terms → "
            f"{len(all_entities)} unique entities"
        )

        return EntitySearchResponse(
            entities=all_entities,
            total_found=len(all_entities)
        )

    except Exception as e:
        logger.error(f"Entity search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/chunks", response_model=ChunkSearchResponse)
async def search_chunks(request: ChunkSearchRequest):
    """
    Find chunks containing specified entities

    Returns Chunk nodes with vector_rowids for SQLite mapping.
    Used by GraphRetriever to get document chunks related to entities.
    """
    try:
        if not request.entity_ids and not request.entity_names:
            raise HTTPException(
                status_code=400,
                detail="Must provide either entity_ids or entity_names"
            )

        client = get_neo4j_client()

        async with client.driver.session(database=client.database) as session:
            if request.entity_ids:
                # Search by entity IDs
                cypher = """
                MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
                WHERE elementId(e) IN $entity_ids
                WITH c, COLLECT(DISTINCT e.text) as matched_entities, COUNT(DISTINCT e) as entity_count
                """
            else:
                # Search by entity names
                cypher = """
                MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
                WHERE e.text IN $entity_names OR e.normalized IN $entity_names
                WITH c, COLLECT(DISTINCT e.text) as matched_entities, COUNT(DISTINCT e) as entity_count
                """

            # Add document info if requested
            if request.include_document_info:
                cypher += """
                OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c)
                RETURN
                    elementId(c) as chunk_id,
                    c.vector_rowid as vector_rowid,
                    c.chunk_index as chunk_index,
                    entity_count,
                    matched_entities,
                    d.url as document_url,
                    d.title as document_title
                ORDER BY entity_count DESC, c.chunk_index ASC
                LIMIT $limit
                """
            else:
                cypher += """
                RETURN
                    elementId(c) as chunk_id,
                    c.vector_rowid as vector_rowid,
                    c.chunk_index as chunk_index,
                    entity_count,
                    matched_entities,
                    null as document_url,
                    null as document_title
                ORDER BY entity_count DESC, c.chunk_index ASC
                LIMIT $limit
                """

            params = {"limit": request.limit}
            if request.entity_ids:
                params["entity_ids"] = request.entity_ids
            else:
                params["entity_names"] = request.entity_names

            result = await session.run(cypher, **params)
            records = await result.data()

            chunks = [ChunkMatch(**record) for record in records]

        logger.info(
            f"Chunk search: {len(request.entity_ids or request.entity_names)} entities → "
            f"{len(chunks)} chunks"
        )

        return ChunkSearchResponse(
            chunks=chunks,
            total_found=len(chunks)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chunk search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/expand/entities", response_model=EntityExpansionResponse)
async def expand_entities(request: EntityExpansionRequest):
    """
    Expand entities through graph relationships

    Discovers related entities via:
    - Direct relationships (if they exist)
    - Co-occurrence in same documents/chunks

    Used by EntityExpander for query enrichment.
    """
    try:
        client = get_neo4j_client()

        expanded = []
        seen_ids = set()

        async with client.driver.session(database=client.database) as session:
            # Strategy: Find entities that co-occur with query entities
            # This works with current schema (Entity-MENTIONED_IN->Chunk<-MENTIONED_IN-Entity)
            cypher = """
            MATCH (e1:Entity)-[:MENTIONED_IN]->(c:Chunk)<-[:MENTIONED_IN]-(e2:Entity)
            WHERE (e1.text IN $entity_names OR e1.normalized IN $entity_names)
              AND e2 <> e1
            WITH e2, COUNT(DISTINCT c) as cooccurrence_count
            WHERE cooccurrence_count >= 2
            RETURN
                elementId(e2) as entity_id,
                e2.text as text,
                e2.normalized as normalized,
                e2.type_primary as type_primary,
                e2.type_full as type_full,
                e2.mention_count as mention_count,
                cooccurrence_count,
                'CO_OCCURS' as relationship_type,
                CASE
                    WHEN cooccurrence_count >= 5 THEN 0.9
                    WHEN cooccurrence_count >= 3 THEN 0.7
                    ELSE 0.5
                END as relationship_confidence,
                1 as path_distance
            ORDER BY cooccurrence_count DESC, e2.mention_count DESC
            LIMIT $max_expansions
            """

            result = await session.run(
                cypher,
                entity_names=request.entity_names,
                max_expansions=request.max_expansions
            )

            records = await result.data()

            for record in records:
                entity_id = record["entity_id"]
                if entity_id not in seen_ids:
                    if record.get("relationship_confidence", 0) >= request.min_confidence:
                        seen_ids.add(entity_id)
                        expanded.append(RelatedEntity(**record))

        logger.info(
            f"Entity expansion: {len(request.entity_names)} → "
            f"{len(expanded)} related entities"
        )

        return EntityExpansionResponse(
            original_entities=request.entity_names,
            expanded_entities=expanded,
            total_discovered=len(expanded)
        )

    except Exception as e:
        logger.error(f"Entity expansion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
