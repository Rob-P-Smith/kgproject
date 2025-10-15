"""
Neo4j Client for Knowledge Graph Storage

Handles all graph database operations including:
- Connection management
- Entity and relationship storage
- Document and chunk management
- Graph queries for retrieval
"""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import asyncio

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, AuthError

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class Neo4jClient:
    """Async Neo4j client for knowledge graph operations"""

    def __init__(self):
        """Initialize Neo4j client"""
        self.driver: Optional[AsyncDriver] = None
        self.uri = settings.NEO4J_URI
        self.user = settings.NEO4J_USER
        self.password = settings.NEO4J_PASSWORD
        self.database = settings.NEO4J_DATABASE
        self._connected = False

    async def connect(self) -> bool:
        """
        Establish connection to Neo4j

        Returns:
            True if connected successfully
        """
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=settings.NEO4J_MAX_CONNECTION_LIFETIME,
                max_connection_pool_size=settings.NEO4J_MAX_CONNECTION_POOL_SIZE,
                connection_timeout=settings.NEO4J_CONNECTION_TIMEOUT
            )

            # Verify connection
            await self.driver.verify_connectivity()

            self._connected = True
            logger.info(f"Connected to Neo4j at {self.uri}")
            return True

        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self._connected = False
            return False

    async def close(self):
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
            self._connected = False
            logger.info("Neo4j connection closed")

    async def health_check(self) -> Dict[str, Any]:
        """
        Check Neo4j health

        Returns:
            Dictionary with status and details
        """
        if not self._connected or not self.driver:
            return {
                "status": "disconnected",
                "message": "Not connected to Neo4j"
            }

        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run("RETURN 1 AS health")
                await result.single()

                return {
                    "status": "connected",
                    "uri": self.uri,
                    "database": self.database
                }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    async def create_document(
        self,
        content_id: int,
        url: str,
        title: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Create or update a Document node

        Args:
            content_id: Unique content identifier from mcpragcrawl4ai
            url: Document URL
            title: Document title
            metadata: Additional metadata

        Returns:
            Neo4j node ID (elementId)
        """
        query = """
        MERGE (d:Document {content_id: $content_id})
        SET d.url = $url,
            d.title = $title,
            d.created_at = COALESCE(d.created_at, datetime()),
            d.updated_at = datetime(),
            d.metadata = $metadata
        RETURN elementId(d) AS node_id
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                query,
                content_id=content_id,
                url=url,
                title=title,
                metadata=metadata
            )
            record = await result.single()
            return record["node_id"]

    async def create_chunk(
        self,
        document_node_id: str,
        vector_rowid: int,
        chunk_index: int,
        char_start: int,
        char_end: int,
        text_preview: str
    ) -> str:
        """
        Create a Chunk node and link to Document

        Args:
            document_node_id: Neo4j elementId of parent Document
            vector_rowid: SQLite content_vectors rowid
            chunk_index: Chunk sequence number
            char_start: Start position in document
            char_end: End position in document
            text_preview: First 200 chars of chunk

        Returns:
            Neo4j node ID (elementId)
        """
        query = """
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
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                query,
                doc_id=document_node_id,
                vector_rowid=vector_rowid,
                chunk_index=chunk_index,
                char_start=char_start,
                char_end=char_end,
                text_preview=text_preview[:200]
            )
            record = await result.single()
            return record["node_id"]

    async def create_entity(
        self,
        text: str,
        normalized: str,
        type_primary: str,
        type_sub1: Optional[str],
        type_sub2: Optional[str],
        type_sub3: Optional[str],
        type_full: str,
        confidence: float
    ) -> str:
        """
        Create or update an Entity node

        Args:
            text: Original entity text
            normalized: Normalized entity name
            type_primary: Primary type (e.g., "Framework")
            type_sub1: First subtype (e.g., "Backend")
            type_sub2: Second subtype (e.g., "Python")
            type_sub3: Third subtype (optional)
            type_full: Full hierarchical type
            confidence: Extraction confidence

        Returns:
            Neo4j node ID (elementId)
        """
        query = """
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
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                query,
                text=text,
                normalized=normalized,
                type_primary=type_primary,
                type_sub1=type_sub1,
                type_sub2=type_sub2,
                type_sub3=type_sub3,
                type_full=type_full,
                confidence=confidence
            )
            record = await result.single()
            return record["node_id"]

    async def link_entity_to_chunk(
        self,
        entity_node_id: str,
        chunk_node_id: str,
        offset_start: int,
        offset_end: int,
        confidence: float,
        context_before: str,
        context_after: str,
        sentence: str
    ):
        """
        Create MENTIONED_IN relationship between Entity and Chunk

        Args:
            entity_node_id: Neo4j elementId of entity
            chunk_node_id: Neo4j elementId of chunk
            offset_start: Character offset in chunk
            offset_end: Character offset in chunk
            confidence: Mention confidence
            context_before: Text before mention
            context_after: Text after mention
            sentence: Sentence containing mention
        """
        query = """
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
        """

        async with self.driver.session(database=self.database) as session:
            await session.run(
                query,
                entity_id=entity_node_id,
                chunk_id=chunk_node_id,
                offset_start=offset_start,
                offset_end=offset_end,
                confidence=confidence,
                context_before=context_before[:100],
                context_after=context_after[:100],
                sentence=sentence[:500]
            )

    async def create_relationship(
        self,
        subject_normalized: str,
        predicate: str,
        object_normalized: str,
        confidence: float,
        context: str
    ):
        """
        Create relationship between two entities

        Args:
            subject_normalized: Normalized subject entity
            predicate: Relationship type
            object_normalized: Normalized object entity
            confidence: Relationship confidence
            context: Supporting text
        """
        # Create dynamic relationship type (uppercase predicate)
        rel_type = predicate.upper().replace(" ", "_")

        query = f"""
        MATCH (s:Entity {{normalized: $subject}})
        MATCH (o:Entity {{normalized: $object}})
        MERGE (s)-[r:{rel_type}]->(o)
        ON CREATE SET
            r.confidence = $confidence,
            r.context = $context,
            r.created_at = datetime(),
            r.occurrence_count = 1
        ON MATCH SET
            r.confidence = (r.confidence * (r.occurrence_count) + $confidence) / (r.occurrence_count + 1),
            r.occurrence_count = r.occurrence_count + 1,
            r.updated_at = datetime()
        """

        async with self.driver.session(database=self.database) as session:
            await session.run(
                query,
                subject=subject_normalized,
                object=object_normalized,
                confidence=confidence,
                context=context[:500]
            )

    async def update_co_occurrence(
        self,
        entity1_normalized: str,
        entity2_normalized: str,
        chunk_rowid: int
    ):
        """
        Track entity co-occurrence in chunks

        Args:
            entity1_normalized: First entity
            entity2_normalized: Second entity
            chunk_rowid: Chunk where they co-occur
        """
        query = """
        MATCH (e1:Entity {normalized: $entity1})
        MATCH (e2:Entity {normalized: $entity2})
        WHERE e1.normalized < e2.normalized  // Ensure consistent direction
        MERGE (e1)-[co:CO_OCCURS_WITH]->(e2)
        ON CREATE SET
            co.count = 1,
            co.chunk_rowids = [$chunk_rowid],
            co.created_at = datetime()
        ON MATCH SET
            co.count = co.count + 1,
            co.chunk_rowids = co.chunk_rowids + $chunk_rowid,
            co.updated_at = datetime()
        """

        # Ensure consistent ordering
        if entity1_normalized > entity2_normalized:
            entity1_normalized, entity2_normalized = entity2_normalized, entity1_normalized

        async with self.driver.session(database=self.database) as session:
            await session.run(
                query,
                entity1=entity1_normalized,
                entity2=entity2_normalized,
                chunk_rowid=chunk_rowid
            )

    async def get_document_stats(self, content_id: int) -> Dict[str, Any]:
        """
        Get statistics for a document

        Args:
            content_id: Content identifier

        Returns:
            Dictionary with document statistics
        """
        query = """
        MATCH (d:Document {content_id: $content_id})
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(c)
        RETURN
            elementId(d) AS doc_id,
            d.url AS url,
            d.title AS title,
            COUNT(DISTINCT c) AS chunk_count,
            COUNT(DISTINCT e) AS entity_count
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, content_id=content_id)
            record = await result.single()

            if not record:
                return None

            return {
                "doc_id": record["doc_id"],
                "url": record["url"],
                "title": record["title"],
                "chunk_count": record["chunk_count"],
                "entity_count": record["entity_count"]
            }

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


# Global instance
_neo4j_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    """Get global Neo4j client instance"""
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    return _neo4j_client
