"""
Pydantic models for kg-service API

Defines request/response models for communication between
mcpragcrawl4ai and kg-service.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime


# ============================================================================
# Request Models
# ============================================================================

class ChunkMetadata(BaseModel):
    """Metadata for a single chunk from mcpragcrawl4ai"""

    vector_rowid: int = Field(
        ...,
        description="SQLite content_vectors rowid",
        gt=0
    )
    chunk_index: int = Field(
        ...,
        description="Sequential chunk number",
        ge=0
    )
    char_start: int = Field(
        ...,
        description="Character position in original markdown",
        ge=0
    )
    char_end: int = Field(
        ...,
        description="Character end position",
        gt=0
    )
    text: str = Field(
        ...,
        description="Actual chunk text",
        min_length=10,
        max_length=10000
    )

    @validator('char_end')
    def end_after_start(cls, v, values):
        """Validate char_end > char_start"""
        if 'char_start' in values and v <= values['char_start']:
            raise ValueError('char_end must be greater than char_start')
        return v

    class Config:
        schema_extra = {
            "example": {
                "vector_rowid": 45001,
                "chunk_index": 0,
                "char_start": 0,
                "char_end": 2500,
                "text": "# FastAPI\n\nFastAPI is a modern web framework..."
            }
        }


class IngestRequest(BaseModel):
    """Request to ingest document for KG processing"""

    content_id: int = Field(
        ...,
        description="mcpragcrawl4ai content ID (primary key)",
        gt=0
    )
    url: str = Field(
        ...,
        description="Source URL",
        max_length=2048
    )
    title: str = Field(
        ...,
        description="Document title",
        max_length=500
    )
    markdown: str = Field(
        ...,
        description="Full document markdown",
        min_length=50,
        max_length=1000000
    )
    chunks: List[ChunkMetadata] = Field(
        ...,
        description="Chunk boundaries with positions",
        min_items=1,
        max_items=1000
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (tags, timestamp, etc.)"
    )

    @validator('chunks')
    def validate_chunks_ordered(cls, v):
        """Ensure chunks are ordered by chunk_index"""
        for i in range(len(v) - 1):
            if v[i].chunk_index >= v[i+1].chunk_index:
                raise ValueError('Chunks must be ordered by chunk_index')
        return v

    @validator('url')
    def validate_url_format(cls, v):
        """Basic URL validation"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

    class Config:
        schema_extra = {
            "example": {
                "content_id": 123,
                "url": "https://docs.fastapi.com",
                "title": "FastAPI Documentation",
                "markdown": "# FastAPI\n\nFastAPI is a modern...",
                "chunks": [
                    {
                        "vector_rowid": 45001,
                        "chunk_index": 0,
                        "char_start": 0,
                        "char_end": 2500,
                        "text": "# FastAPI\n\nFastAPI is..."
                    }
                ],
                "metadata": {
                    "tags": "python,api,web",
                    "timestamp": "2025-10-15T12:00:00Z"
                }
            }
        }


# ============================================================================
# Response Models
# ============================================================================

class EntityAppearance(BaseModel):
    """Entity appearance in a specific chunk"""

    vector_rowid: int = Field(..., description="Chunk vector_rowid")
    chunk_index: int = Field(..., description="Chunk index")
    offset_start: int = Field(..., description="Start position within chunk")
    offset_end: int = Field(..., description="End position within chunk")

    class Config:
        schema_extra = {
            "example": {
                "vector_rowid": 45001,
                "chunk_index": 0,
                "offset_start": 342,
                "offset_end": 349
            }
        }


class ExtractedEntity(BaseModel):
    """Entity extracted from document"""

    text: str = Field(..., description="Entity text")
    normalized: str = Field(..., description="Normalized text (lowercase)")

    # Type hierarchy
    type_primary: str = Field(..., description="Primary type (e.g., Framework)")
    type_sub1: Optional[str] = Field(None, description="Subtype level 1 (e.g., Backend)")
    type_sub2: Optional[str] = Field(None, description="Subtype level 2 (e.g., Python)")
    type_sub3: Optional[str] = Field(None, description="Subtype level 3 (optional)")
    type_full: str = Field(..., description="Full hierarchical type")

    # Metadata
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    neo4j_node_id: str = Field(..., description="Neo4j node ID")

    # Context
    context_before: str = Field("", description="Text before entity")
    context_after: str = Field("", description="Text after entity")
    sentence: str = Field("", description="Full sentence containing entity")

    # Chunk mappings
    chunk_appearances: List[EntityAppearance] = Field(
        ...,
        description="Chunks where this entity appears"
    )
    spans_multiple_chunks: bool = Field(
        ...,
        description="True if entity appears in multiple chunks"
    )

    class Config:
        schema_extra = {
            "example": {
                "text": "FastAPI",
                "normalized": "fastapi",
                "type_primary": "Framework",
                "type_sub1": "Backend",
                "type_sub2": "Python",
                "type_sub3": None,
                "type_full": "Framework::Backend::Python",
                "confidence": 0.95,
                "neo4j_node_id": "4:entity:789",
                "context_before": "modern web ",
                "context_after": " for building",
                "sentence": "FastAPI is a modern web framework for building APIs.",
                "chunk_appearances": [
                    {
                        "vector_rowid": 45001,
                        "chunk_index": 0,
                        "offset_start": 342,
                        "offset_end": 349
                    }
                ],
                "spans_multiple_chunks": False
            }
        }


class ExtractedRelationship(BaseModel):
    """Relationship between two entities"""

    subject_text: str = Field(..., description="Subject entity text")
    subject_neo4j_id: str = Field(..., description="Subject Neo4j node ID")
    predicate: str = Field(..., description="Relationship type (uses, implements, etc.)")
    object_text: str = Field(..., description="Object entity text")
    object_neo4j_id: str = Field(..., description="Object Neo4j node ID")

    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    context: str = Field(..., description="Sentence where relationship found")

    neo4j_relationship_id: Optional[str] = Field(None, description="Neo4j relationship ID")

    # Chunk mapping
    spans_chunks: bool = Field(
        ...,
        description="True if entities in different chunks"
    )
    chunk_rowids: List[int] = Field(
        ...,
        description="Chunks involved in this relationship"
    )

    class Config:
        schema_extra = {
            "example": {
                "subject_text": "FastAPI",
                "subject_neo4j_id": "4:entity:789",
                "predicate": "uses",
                "object_text": "Pydantic",
                "object_neo4j_id": "4:entity:790",
                "confidence": 0.88,
                "context": "FastAPI uses Pydantic for data validation",
                "neo4j_relationship_id": "5:rel:101",
                "spans_chunks": False,
                "chunk_rowids": [45001]
            }
        }


class ProcessingSummary(BaseModel):
    """Summary statistics of processing"""

    entities_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of entities by primary type"
    )
    relationships_by_predicate: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of relationships by predicate"
    )
    chunks_with_entities: int = Field(
        0,
        description="Number of chunks containing entities"
    )
    avg_entities_per_chunk: float = Field(
        0.0,
        description="Average entities per chunk"
    )


class IngestResponse(BaseModel):
    """Response after successful ingestion"""

    success: bool = Field(..., description="Processing success status")
    content_id: int = Field(..., description="mcpragcrawl4ai content ID")
    neo4j_document_id: str = Field(..., description="Neo4j Document node ID")

    entities_extracted: int = Field(..., description="Total entities extracted")
    relationships_extracted: int = Field(..., description="Total relationships extracted")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")

    entities: List[ExtractedEntity] = Field(
        ...,
        description="Detailed entity data"
    )
    relationships: List[ExtractedRelationship] = Field(
        ...,
        description="Detailed relationship data"
    )

    summary: ProcessingSummary = Field(
        ...,
        description="Processing statistics"
    )

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "content_id": 123,
                "neo4j_document_id": "4:doc:456",
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
                        "type_sub3": None,
                        "type_full": "Framework::Backend::Python",
                        "confidence": 0.95,
                        "neo4j_node_id": "4:entity:789",
                        "chunk_appearances": [
                            {
                                "vector_rowid": 45001,
                                "chunk_index": 0,
                                "offset_start": 342,
                                "offset_end": 349
                            }
                        ],
                        "spans_multiple_chunks": False
                    }
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
                        "spans_chunks": False,
                        "chunk_rowids": [45001]
                    }
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
                    },
                    "chunks_with_entities": 18,
                    "avg_entities_per_chunk": 4.8
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response"""

    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    content_id: Optional[int] = Field(None, description="Content ID if available")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": "Failed to extract entities: Model timeout",
                "error_type": "ProcessingError",
                "content_id": 123,
                "timestamp": "2025-10-15T12:00:00Z"
            }
        }


# ============================================================================
# Health & Status Models
# ============================================================================

class HealthStatus(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status (healthy, degraded, unhealthy)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    services: Dict[str, str] = Field(
        ...,
        description="Status of dependent services"
    )

    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-10-15T12:00:00Z",
                "services": {
                    "neo4j": "connected",
                    "vllm": "connected",
                    "gliner": "loaded"
                },
                "version": "1.0.0",
                "uptime_seconds": 3600.5
            }
        }


class ServiceStats(BaseModel):
    """Service statistics"""

    total_documents_processed: int = Field(0, description="Total documents processed")
    total_entities_extracted: int = Field(0, description="Total entities extracted")
    total_relationships_extracted: int = Field(0, description="Total relationships extracted")

    avg_processing_time_ms: float = Field(0.0, description="Average processing time")
    last_processed_at: Optional[datetime] = Field(None, description="Last processing timestamp")

    queue_size: int = Field(0, description="Current processing queue size")
    failed_count: int = Field(0, description="Failed processing count")

    class Config:
        schema_extra = {
            "example": {
                "total_documents_processed": 523,
                "total_entities_extracted": 45234,
                "total_relationships_extracted": 12456,
                "avg_processing_time_ms": 2341.5,
                "last_processed_at": "2025-10-15T12:00:00Z",
                "queue_size": 3,
                "failed_count": 2
            }
        }
