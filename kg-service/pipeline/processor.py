"""
Knowledge Graph Processing Pipeline

Orchestrates the complete pipeline:
1. Entity extraction (GLiNER)
2. Relationship extraction (vLLM)
3. Chunk mapping
4. Neo4j storage

This is the main processor invoked by the API.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from extractors.entity_extractor import get_entity_extractor
from extractors.relation_extractor import get_relation_extractor
from pipeline.chunk_mapper import get_chunk_mapper
from storage.neo4j_client import get_neo4j_client
from storage.schema import initialize_graph_schema
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class KGProcessor:
    """Main knowledge graph processing pipeline"""

    def __init__(self):
        """Initialize the KG processor"""
        self.entity_extractor = get_entity_extractor()
        self.relation_extractor = get_relation_extractor()
        self.chunk_mapper = get_chunk_mapper()
        self.neo4j_client = get_neo4j_client()
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """
        Initialize all components

        Returns:
            True if initialization successful
        """
        async with self._init_lock:
            if self._initialized:
                return True

            logger.info("Initializing KG Processor...")

            try:
                # Connect to Neo4j
                connected = await self.neo4j_client.connect()
                if not connected:
                    logger.error("Failed to connect to Neo4j")
                    return False

                # Initialize schema
                schema_results = await initialize_graph_schema(self.neo4j_client)
                logger.info(
                    f"Schema initialized: {schema_results['constraints_created']} constraints, "
                    f"{schema_results['indexes_created']} indexes"
                )

                self._initialized = True
                logger.info("✓ KG Processor initialized successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to initialize KG Processor: {e}")
                return False

    async def shutdown(self):
        """Shutdown and cleanup resources"""
        logger.info("Shutting down KG Processor...")
        await self.neo4j_client.close()
        self._initialized = False

    async def process_document(
        self,
        content_id: int,
        url: str,
        title: str,
        markdown: str,
        chunks: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a complete document through the KG pipeline

        Args:
            content_id: Unique content identifier
            url: Document URL
            title: Document title
            markdown: Full markdown content
            chunks: List of chunks with boundaries
            metadata: Optional metadata

        Returns:
            Dictionary with processing results
        """
        if not self._initialized:
            await self.initialize()

        start_time = datetime.now()
        logger.info(f"Processing document: {url}")

        try:
            # Step 1: Extract entities from full document
            logger.info("Step 1: Extracting entities with GLiNER...")
            entities = self.entity_extractor.extract(markdown)
            logger.info(f"✓ Extracted {len(entities)} entities")

            # Step 2: Extract relationships
            logger.info("Step 2: Extracting relationships with vLLM...")
            relationships = await self.relation_extractor.extract_relationships(
                markdown,
                entities
            )
            logger.info(f"✓ Extracted {len(relationships)} relationships")

            # Step 3: Map entities to chunks
            logger.info("Step 3: Mapping entities and relationships to chunks...")
            mapped_entities = self.chunk_mapper.map_entities_to_chunks(
                entities,
                chunks
            )

            mapped_relationships = self.chunk_mapper.map_relationships_to_chunks(
                relationships,
                mapped_entities,
                chunks
            )
            logger.info("✓ Chunk mapping complete")

            # Step 4: Store in Neo4j
            logger.info("Step 4: Storing in Neo4j...")
            neo4j_result = await self._store_in_neo4j(
                content_id=content_id,
                url=url,
                title=title,
                metadata=metadata or {},
                chunks=chunks,
                entities=mapped_entities,
                relationships=mapped_relationships
            )
            logger.info("✓ Neo4j storage complete")

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Generate summary
            summary = self.chunk_mapper.generate_mapping_summary(
                mapped_entities,
                mapped_relationships,
                chunks
            )

            # Build response
            result = {
                "success": True,
                "content_id": content_id,
                "neo4j_document_id": neo4j_result["document_node_id"],
                "entities_extracted": len(mapped_entities),
                "relationships_extracted": len(mapped_relationships),
                "processing_time_ms": round(processing_time, 2),
                "entities": self._format_entities_for_response(mapped_entities),
                "relationships": self._format_relationships_for_response(mapped_relationships),
                "summary": summary
            }

            logger.info(
                f"✓ Processing complete: Entities: {len(mapped_entities)}, "
                f"Relationships: {len(mapped_relationships)}, "
                f"Time: {processing_time:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Document processing failed: {e}", exc_info=True)
            raise

    async def _store_in_neo4j(
        self,
        content_id: int,
        url: str,
        title: str,
        metadata: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Store all data in Neo4j graph

        Args:
            content_id: Content identifier
            url: Document URL
            title: Document title
            metadata: Document metadata
            chunks: Chunk list
            entities: Mapped entities
            relationships: Mapped relationships

        Returns:
            Dictionary with Neo4j node IDs
        """
        # Create Document node
        doc_node_id = await self.neo4j_client.create_document(
            content_id=content_id,
            url=url,
            title=title,
            metadata=metadata
        )

        # Create Chunk nodes and build chunk_id map
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

        # Create Entity nodes and build entity_id map
        entity_node_map = {}  # normalized -> node_id
        for entity in entities:
            # Debug: log first entity to check type fields
            if not entity_node_map:
                logger.info(f"DEBUG first entity: text={entity.get('text')}, type_primary={entity.get('type_primary')}, type_full={entity.get('type_full')}")

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

            # Link entity to chunks where it appears
            for appearance in entity.get("chunk_appearances", []):
                chunk_node_id = chunk_node_map.get(appearance["vector_rowid"])
                if chunk_node_id:
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

        # Create semantic relationships
        for rel in relationships:
            await self.neo4j_client.create_relationship(
                subject_normalized=rel["subject_normalized"],
                predicate=rel["predicate"],
                object_normalized=rel["object_normalized"],
                confidence=rel["confidence"],
                context=rel["context"]
            )

        # Update co-occurrence relationships - DISABLED (creates too many relationships)
        # await self._update_co_occurrences(entities, chunks)

        return {
            "document_node_id": doc_node_id,
            "chunk_count": len(chunk_node_map),
            "entity_count": len(entity_node_map),
            "relationship_count": len(relationships)
        }

    async def _update_co_occurrences(
        self,
        entities: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]]
    ):
        """
        Update entity co-occurrence relationships

        Args:
            entities: List of entities with chunk appearances
            chunks: List of chunks
        """
        # Group entities by chunk
        chunk_entities = {}  # vector_rowid -> [entity_normalized, ...]
        for entity in entities:
            for appearance in entity.get("chunk_appearances", []):
                rowid = appearance["vector_rowid"]
                if rowid not in chunk_entities:
                    chunk_entities[rowid] = []
                chunk_entities[rowid].append(entity["normalized"])

        # Update co-occurrences for each chunk
        for rowid, entity_list in chunk_entities.items():
            if len(entity_list) < 2:
                continue

            # Create co-occurrence pairs
            for i, entity1 in enumerate(entity_list):
                for entity2 in entity_list[i + 1:]:
                    await self.neo4j_client.update_co_occurrence(
                        entity1_normalized=entity1,
                        entity2_normalized=entity2,
                        chunk_rowid=rowid
                    )

    def _format_entities_for_response(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format entities for API response"""
        formatted = []
        for entity in entities:
            formatted.append({
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
            })
        return formatted

    def _format_relationships_for_response(
        self,
        relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format relationships for API response"""
        formatted = []
        for rel in relationships:
            formatted.append({
                "subject_text": rel["subject_text"],
                "subject_normalized": rel["subject_normalized"],
                "predicate": rel["predicate"],
                "object_text": rel["object_text"],
                "object_normalized": rel["object_normalized"],
                "confidence": rel["confidence"],
                "context": rel["context"],
                "spans_chunks": rel.get("spans_chunks", False),
                "chunk_rowids": rel.get("chunk_rowids", [])
            })
        return formatted


# Global instance
_kg_processor: Optional[KGProcessor] = None


def get_kg_processor() -> KGProcessor:
    """Get global KG processor instance"""
    global _kg_processor
    if _kg_processor is None:
        _kg_processor = KGProcessor()
    return _kg_processor
