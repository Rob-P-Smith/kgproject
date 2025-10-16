"""
FastAPI server for kg-service

Receives markdown documents from mcpragcrawl4ai,
processes entities and relationships, stores in Neo4j.
"""

import time
import logging
from datetime import datetime
from typing import Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import (
    IngestRequest,
    IngestResponse,
    ErrorResponse,
    HealthStatus,
    ServiceStats
)
from config import settings, validate_settings

# Import processors (will be created)
from pipeline.processor import KGProcessor
from clients.vllm_client import get_vllm_client, close_vllm_client
from extractors.entity_extractor import get_entity_extractor

logger = logging.getLogger(__name__)

# Global state
service_start_time = time.time()
processing_stats = {
    "total_processed": 0,
    "total_entities": 0,
    "total_relationships": 0,
    "total_processing_time_ms": 0,
    "failed_count": 0,
    "last_processed_at": None
}

# Global processor instance
kg_processor: KGProcessor = None


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic
    """
    global kg_processor

    # Startup
    logger.info("=" * 60)
    logger.info("Starting kg-service")
    logger.info("=" * 60)

    # Validate configuration
    if not validate_settings():
        logger.error("Configuration validation failed!")
        raise RuntimeError("Invalid configuration")

    logger.info(f"Service: {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")
    logger.info(f"API: {settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"Neo4j: {settings.NEO4J_URI}")
    logger.info(f"vLLM: {settings.VLLM_BASE_URL}")
    logger.info(f"GLiNER: {settings.GLINER_MODEL}")

    try:
        # Initialize components
        logger.info("Initializing KG processor...")
        kg_processor = KGProcessor()

        # Initialize Neo4j connection
        if not await kg_processor.initialize():
            logger.error("Failed to initialize KG processor (Neo4j connection failed)")
            raise RuntimeError("KG processor initialization failed")

        # Wait for vLLM to be available (async, non-blocking)
        logger.info("Checking vLLM availability...")
        vllm_client = await get_vllm_client()
        if not await vllm_client.ensure_model():
            logger.warning(
                "vLLM not immediately available - will retry on first request"
            )
        else:
            logger.info(f"✓ vLLM ready: {vllm_client.model_name}")

        logger.info("✓ kg-service ready")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield  # Server runs

    # Shutdown
    logger.info("Shutting down kg-service...")
    await close_vllm_client()
    if kg_processor:
        await kg_processor.shutdown()
    logger.info("✓ Shutdown complete")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Knowledge Graph Service",
    description="Entity and relationship extraction for RAG systems",
    version=settings.SERVICE_VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Middleware
# ============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Log
    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.2f}ms"
    )

    # Add timing header
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"

    return response


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error=exc.detail,
            error_type="HTTPException",
            timestamp=datetime.utcnow()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error=str(exc),
            error_type=type(exc).__name__,
            timestamp=datetime.utcnow()
        ).dict()
    )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def health_check():
    """
    Health check endpoint

    Returns service health status and dependent service connectivity
    """
    uptime = time.time() - service_start_time

    # Check dependent services
    services_status = {}

    # Check Neo4j
    try:
        if kg_processor and kg_processor.neo4j_client:
            await kg_processor.neo4j_client.driver.verify_connectivity()
            services_status["neo4j"] = "connected"
        else:
            services_status["neo4j"] = "not_initialized"
    except Exception as e:
        services_status["neo4j"] = f"error: {str(e)[:50]}"

    # Check vLLM
    try:
        vllm_client = await get_vllm_client()
        if await vllm_client.health_check():
            services_status["vllm"] = f"connected ({vllm_client.model_name or 'discovering'})"
        else:
            services_status["vllm"] = "unhealthy"
    except Exception as e:
        services_status["vllm"] = f"error: {str(e)[:50]}"

    # Check GLiNER
    try:
        extractor = get_entity_extractor()
        if extractor and extractor.model:
            services_status["gliner"] = "loaded"
        else:
            services_status["gliner"] = "not_loaded"
    except Exception as e:
        services_status["gliner"] = f"error: {str(e)[:50]}"

    # Determine overall status
    if all(
        status in ["connected", "loaded"] or status.startswith("connected (")
        for status in services_status.values()
    ):
        overall_status = "healthy"
    elif any("error" in status for status in services_status.values()):
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"

    return HealthStatus(
        status=overall_status,
        timestamp=datetime.utcnow(),
        services=services_status,
        version=settings.SERVICE_VERSION,
        uptime_seconds=uptime
    )


@app.get("/stats", response_model=ServiceStats, tags=["Stats"])
async def get_stats():
    """
    Get service statistics

    Returns processing metrics and counts
    """
    avg_time = 0.0
    if processing_stats["total_processed"] > 0:
        avg_time = (
            processing_stats["total_processing_time_ms"] /
            processing_stats["total_processed"]
        )

    return ServiceStats(
        total_documents_processed=processing_stats["total_processed"],
        total_entities_extracted=processing_stats["total_entities"],
        total_relationships_extracted=processing_stats["total_relationships"],
        avg_processing_time_ms=avg_time,
        last_processed_at=processing_stats["last_processed_at"],
        queue_size=0,  # No queue in this service
        failed_count=processing_stats["failed_count"]
    )


@app.post(
    "/api/v1/ingest",
    response_model=IngestResponse,
    tags=["Processing"],
    summary="Ingest document for KG processing"
)
async def ingest_document(request: IngestRequest):
    """
    Ingest document from mcpragcrawl4ai for entity/relationship extraction

    This is the main endpoint called by mcpragcrawl4ai after it has:
    1. Crawled and cleaned a document
    2. Chunked the content
    3. Generated embeddings
    4. Stored in SQLite

    The kg-service will:
    1. Extract entities using GLiNER (300+ types)
    2. Extract relationships using vLLM
    3. Map entities/relationships to chunks
    4. Store in Neo4j graph database
    5. Return results for mcpragcrawl4ai to store in SQLite

    Args:
        request: IngestRequest with full markdown and chunk boundaries

    Returns:
        IngestResponse with extracted entities, relationships, and Neo4j IDs

    Raises:
        HTTPException: If processing fails
    """

    start_time = time.time()

    logger.info("=" * 60)
    logger.info(f"Processing document: {request.url}")
    logger.info(f"Content ID: {request.content_id}")
    logger.info(f"Title: {request.title}")
    logger.info(f"Markdown length: {len(request.markdown)} chars")
    logger.info(f"Chunks: {len(request.chunks)}")
    logger.info("=" * 60)

    try:
        # Check if processor is initialized
        if not kg_processor:
            raise HTTPException(
                status_code=503,
                detail="KG processor not initialized"
            )

        # Process document
        result = await kg_processor.process_document(
            content_id=request.content_id,
            url=request.url,
            title=request.title,
            markdown=request.markdown,
            chunks=[chunk.dict() for chunk in request.chunks],
            metadata=request.metadata
        )

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Update stats
        processing_stats["total_processed"] += 1
        processing_stats["total_entities"] += result["entities_extracted"]
        processing_stats["total_relationships"] += result["relationships_extracted"]
        processing_stats["total_processing_time_ms"] += processing_time_ms
        processing_stats["last_processed_at"] = datetime.utcnow()

        # Build response
        response = IngestResponse(
            success=True,
            content_id=request.content_id,
            neo4j_document_id=result["neo4j_document_id"],
            entities_extracted=result["entities_extracted"],
            relationships_extracted=result["relationships_extracted"],
            processing_time_ms=processing_time_ms,
            entities=result["entities"],
            relationships=result["relationships"],
            summary=result["summary"]
        )

        logger.info(f"✓ Processing complete:")
        logger.info(f"  - Entities: {result['entities_extracted']}")
        logger.info(f"  - Relationships: {result['relationships_extracted']}")
        logger.info(f"  - Time: {processing_time_ms}ms")
        logger.info("=" * 60)

        return response

    except Exception as e:
        logger.error(f"✗ Processing failed: {e}", exc_info=True)

        # Update error stats
        processing_stats["failed_count"] += 1

        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


@app.get("/api/v1/model-info", tags=["Info"])
async def get_model_info():
    """
    Get information about loaded models

    Returns details about GLiNER and vLLM models
    """
    info = {
        "gliner": {
            "model": settings.GLINER_MODEL,
            "threshold": settings.GLINER_THRESHOLD,
            "status": "unknown"
        },
        "vllm": {
            "base_url": settings.VLLM_BASE_URL,
            "model_name": None,
            "status": "unknown"
        }
    }

    # Get GLiNER info
    try:
        extractor = get_entity_extractor()
        if extractor and extractor.model:
            info["gliner"]["status"] = "loaded"
            info["gliner"]["entity_types_count"] = len(extractor.entity_types)
    except Exception as e:
        info["gliner"]["status"] = f"error: {str(e)}"

    # Get vLLM info
    try:
        vllm_client = await get_vllm_client()
        model_info = await vllm_client.get_model_info()
        if model_info:
            info["vllm"]["model_name"] = model_info.get("id")
            info["vllm"]["status"] = "connected"
        else:
            info["vllm"]["status"] = "unavailable"
    except Exception as e:
        info["vllm"]["status"] = f"error: {str(e)}"

    return info


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
