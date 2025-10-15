"""
Chunk Mapper - Maps entities and relationships to document chunks

This module maps entities and relationships extracted from full documents
to specific chunks using verified character boundaries. This ensures precise
retrieval while maintaining context during extraction.
"""

import logging
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChunkBoundary:
    """Chunk boundary information"""
    vector_rowid: int
    chunk_index: int
    char_start: int
    char_end: int
    text: str


@dataclass
class EntityChunkMapping:
    """Mapping of entity to chunks where it appears"""
    entity_text: str
    entity_normalized: str
    entity_type: str
    confidence: float
    chunk_appearances: List[Dict[str, Any]]  # List of {vector_rowid, chunk_index, offset_start, offset_end}
    spans_multiple_chunks: bool


@dataclass
class RelationshipChunkMapping:
    """Mapping of relationship to chunks"""
    subject_text: str
    predicate: str
    object_text: str
    confidence: float
    spans_chunks: bool  # True if subject and object in different chunks
    chunk_rowids: List[int]  # All chunks involved
    primary_chunk_rowid: int  # Chunk where relationship is most relevant


class ChunkMapper:
    """Maps entities and relationships to document chunks"""

    def __init__(self):
        """Initialize chunk mapper"""
        self.overlap_threshold = 10  # Minimum overlap characters to count as "in chunk"

    def map_entities_to_chunks(
        self,
        entities: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Map entities to chunks based on character positions

        Args:
            entities: List of entities with positions in full document
            chunks: List of chunks with char_start/char_end boundaries

        Returns:
            List of entities enriched with chunk_appearances data
        """
        if not chunks:
            logger.warning("No chunks provided for mapping")
            return entities

        # Parse chunk boundaries
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

        logger.info(f"Mapping {len(entities)} entities to {len(chunk_boundaries)} chunks")

        enriched_entities = []
        for entity in entities:
            # Get all occurrences of entity in document
            occurrences = entity.get("occurrences", [])

            # If no occurrences provided, create one from entity position
            if not occurrences:
                start_pos = entity.get("start_pos", 0)
                end_pos = entity.get("end_pos", start_pos + len(entity["text"]))
                occurrences = [{
                    "start": start_pos,
                    "end": end_pos,
                    "context": entity.get("sentence", "")
                }]

            # Map each occurrence to chunks
            chunk_appearances = []
            seen_chunks = set()

            for occurrence in occurrences:
                occ_start = occurrence["start"]
                occ_end = occurrence["end"]

                # Find overlapping chunks
                for chunk in chunk_boundaries:
                    overlap = self._calculate_overlap(
                        occ_start, occ_end,
                        chunk.char_start, chunk.char_end
                    )

                    if overlap >= self.overlap_threshold:
                        chunk_key = (chunk.vector_rowid, chunk.chunk_index)
                        if chunk_key not in seen_chunks:
                            # Calculate offset within chunk
                            offset_start = max(0, occ_start - chunk.char_start)
                            offset_end = min(
                                len(chunk.text),
                                occ_end - chunk.char_start
                            )

                            chunk_appearances.append({
                                "vector_rowid": chunk.vector_rowid,
                                "chunk_index": chunk.chunk_index,
                                "offset_start": offset_start,
                                "offset_end": offset_end
                            })
                            seen_chunks.add(chunk_key)

            # Enrich entity with chunk data
            enriched_entity = entity.copy()
            enriched_entity["chunk_appearances"] = chunk_appearances
            enriched_entity["spans_multiple_chunks"] = len(chunk_appearances) > 1
            enriched_entity["num_chunks"] = len(chunk_appearances)

            enriched_entities.append(enriched_entity)

        # Log statistics
        total_appearances = sum(len(e["chunk_appearances"]) for e in enriched_entities)
        multi_chunk = sum(1 for e in enriched_entities if e["spans_multiple_chunks"])

        logger.info(
            f"Mapped entities: {total_appearances} total appearances, "
            f"{multi_chunk} entities span multiple chunks"
        )

        return enriched_entities

    def map_relationships_to_chunks(
        self,
        relationships: List[Dict[str, Any]],
        entity_chunk_map: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Map relationships to chunks based on entity positions

        Args:
            relationships: List of relationships between entities
            entity_chunk_map: Entities with chunk_appearances data
            chunks: List of chunk boundaries

        Returns:
            List of relationships enriched with chunk mapping data
        """
        if not chunks:
            logger.warning("No chunks provided for relationship mapping")
            return relationships

        # Build entity lookup by normalized name
        entity_lookup = {}
        for entity in entity_chunk_map:
            entity_lookup[entity["normalized"]] = entity

        logger.info(f"Mapping {len(relationships)} relationships to chunks")

        enriched_relationships = []
        for rel in relationships:
            subject_norm = rel["subject_normalized"]
            object_norm = rel["object_normalized"]

            # Get entities
            subject_entity = entity_lookup.get(subject_norm)
            object_entity = entity_lookup.get(object_norm)

            if not subject_entity or not object_entity:
                logger.warning(
                    f"Entity not found for relationship: {rel['subject_text']} "
                    f"-> {rel['object_text']}"
                )
                continue

            # Get chunk appearances
            subject_chunks = set(
                app["vector_rowid"]
                for app in subject_entity.get("chunk_appearances", [])
            )
            object_chunks = set(
                app["vector_rowid"]
                for app in object_entity.get("chunk_appearances", [])
            )

            # Find all chunks involved
            all_chunks = subject_chunks | object_chunks
            chunk_rowids = sorted(list(all_chunks))

            # Check if relationship spans chunks
            spans_chunks = not bool(subject_chunks & object_chunks)

            # Determine primary chunk (where both entities appear, or closest)
            primary_chunk_rowid = self._find_primary_chunk(
                subject_entity,
                object_entity,
                chunks
            )

            # Enrich relationship
            enriched_rel = rel.copy()
            enriched_rel["spans_chunks"] = spans_chunks
            enriched_rel["chunk_rowids"] = chunk_rowids
            enriched_rel["primary_chunk_rowid"] = primary_chunk_rowid
            enriched_rel["num_chunks_involved"] = len(chunk_rowids)

            enriched_relationships.append(enriched_rel)

        # Log statistics
        cross_chunk = sum(1 for r in enriched_relationships if r["spans_chunks"])
        logger.info(
            f"Mapped relationships: {len(enriched_relationships)} total, "
            f"{cross_chunk} span multiple chunks"
        )

        return enriched_relationships

    def _calculate_overlap(
        self,
        start1: int,
        end1: int,
        start2: int,
        end2: int
    ) -> int:
        """Calculate character overlap between two ranges"""
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        return max(0, overlap_end - overlap_start)

    def _find_primary_chunk(
        self,
        subject_entity: Dict[str, Any],
        object_entity: Dict[str, Any],
        chunks: List[Dict[str, Any]]
    ) -> Optional[int]:
        """
        Find the primary chunk for a relationship

        Priority:
        1. Chunk where both entities appear
        2. Chunk closest to both entities
        3. First chunk containing subject
        """
        subject_chunks = {
            app["vector_rowid"]: app
            for app in subject_entity.get("chunk_appearances", [])
        }
        object_chunks = {
            app["vector_rowid"]: app
            for app in object_entity.get("chunk_appearances", [])
        }

        # Check for shared chunks
        shared = set(subject_chunks.keys()) & set(object_chunks.keys())
        if shared:
            return min(shared)  # Return first shared chunk

        # Find closest chunks
        if subject_chunks and object_chunks:
            min_distance = float('inf')
            primary_rowid = None

            for subj_rowid in subject_chunks:
                for obj_rowid in object_chunks:
                    distance = abs(
                        self._get_chunk_index(subj_rowid, chunks) -
                        self._get_chunk_index(obj_rowid, chunks)
                    )
                    if distance < min_distance:
                        min_distance = distance
                        # Use the earlier chunk
                        primary_rowid = min(subj_rowid, obj_rowid)

            return primary_rowid

        # Fallback to first chunk with subject
        if subject_chunks:
            return min(subject_chunks.keys())

        # Last resort: first chunk with object
        if object_chunks:
            return min(object_chunks.keys())

        return None

    def _get_chunk_index(
        self,
        vector_rowid: int,
        chunks: List[Dict[str, Any]]
    ) -> int:
        """Get chunk index from vector_rowid"""
        for chunk in chunks:
            if chunk["vector_rowid"] == vector_rowid:
                return chunk["chunk_index"]
        return 0

    def generate_mapping_summary(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate summary statistics for chunk mapping

        Args:
            entities: Mapped entities
            relationships: Mapped relationships
            chunks: Chunk list

        Returns:
            Dictionary with mapping statistics
        """
        total_entity_appearances = sum(
            len(e.get("chunk_appearances", []))
            for e in entities
        )
        multi_chunk_entities = sum(
            1 for e in entities
            if e.get("spans_multiple_chunks", False)
        )
        cross_chunk_relationships = sum(
            1 for r in relationships
            if r.get("spans_chunks", False)
        )

        # Find chunks with most entities
        chunk_entity_count = {}
        for entity in entities:
            for app in entity.get("chunk_appearances", []):
                rowid = app["vector_rowid"]
                chunk_entity_count[rowid] = chunk_entity_count.get(rowid, 0) + 1

        chunks_with_entities = len(chunk_entity_count)
        avg_entities_per_chunk = (
            total_entity_appearances / chunks_with_entities
            if chunks_with_entities > 0 else 0
        )

        return {
            "total_chunks": len(chunks),
            "chunks_with_entities": chunks_with_entities,
            "total_entity_appearances": total_entity_appearances,
            "unique_entities": len(entities),
            "multi_chunk_entities": multi_chunk_entities,
            "avg_entities_per_chunk": round(avg_entities_per_chunk, 2),
            "total_relationships": len(relationships),
            "cross_chunk_relationships": cross_chunk_relationships,
            "chunk_entity_distribution": dict(sorted(
                chunk_entity_count.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])  # Top 10 chunks by entity count
        }


# Global instance
_chunk_mapper: Optional[ChunkMapper] = None


def get_chunk_mapper() -> ChunkMapper:
    """Get global chunk mapper instance"""
    global _chunk_mapper
    if _chunk_mapper is None:
        _chunk_mapper = ChunkMapper()
    return _chunk_mapper
