---
layout: default
title: Knowledge Graph Service
---

# Knowledge Graph Service Documentation

## Overview

The Knowledge Graph Service (kg-service) is a production-grade microservice designed to extract entities and semantic relationships from documents and persist them in a Neo4j graph database. It serves as the knowledge graph backend for RAG (Retrieval-Augmented Generation) systems, enabling graph-enhanced semantic search and entity-centric retrieval.

## System Architecture

The service implements a four-stage processing pipeline:

1. **Entity Extraction**: GLiNER-based NER with hierarchical type classification supporting 300+ entity types
2. **Relationship Extraction**: vLLM-powered semantic relationship identification between entities
3. **Chunk Mapping**: Precise mapping of extracted knowledge to document chunk boundaries
4. **Graph Storage**: Persistent storage in Neo4j with optimized schema for retrieval

## Core Capabilities

### Entity Recognition
- Hierarchical entity taxonomy with three levels of classification
- Confidence-based filtering and deduplication
- Context-aware mention extraction with sentence boundaries
- Support for entities spanning multiple document chunks

### Relationship Extraction
- LLM-based semantic relationship identification across eight relationship categories
- Support for cross-chunk relationships
- Confidence scoring and validation
- Co-occurrence tracking for entity proximity analysis

### Graph Storage
- Neo4j-based knowledge graph with optimized schema
- Bidirectional linking between vector chunks (SQLite) and graph nodes (Neo4j)
- Document provenance tracking
- Efficient querying for entity expansion and chunk retrieval

### Integration Points
- RESTful API compatible with mcpragcrawl4ai and other RAG systems
- Asynchronous processing with configurable concurrency
- Health monitoring and statistics endpoints
- Search endpoints for entity discovery and chunk retrieval

## Documentation Structure

### [Architecture Overview](architecture.md)
Complete system architecture, data flow, and component interactions.

### [API Layer](api.md)
FastAPI server implementation, request/response models, and endpoint specifications.

### [Extractors Module](extractors.md)
Entity and relationship extraction engines, including GLiNER and vLLM integration.

### [Pipeline Module](pipeline.md)
Processing orchestration, chunk mapping, and workflow coordination.

### [Storage Module](storage.md)
Neo4j client implementation, schema management, and graph operations.

### [Configuration & Clients](configuration.md)
Service configuration, external client interfaces, and environment management.

## Quick Start

### Prerequisites
- Python 3.9+
- Neo4j 5.x database
- vLLM inference server with LLM model
- 8GB+ RAM for GLiNER model

### Service Startup
The service initializes automatically on startup:
- Validates configuration parameters
- Connects to Neo4j and initializes graph schema
- Loads GLiNER entity extraction model
- Establishes connection to vLLM server
- Starts FastAPI server on configured port

### API Integration
External systems interact via RESTful endpoints:
- `POST /api/v1/ingest`: Submit documents for processing
- `POST /api/v1/search/entities`: Search for entities by text
- `POST /api/v1/search/chunks`: Retrieve chunks containing entities
- `POST /api/v1/expand/entities`: Discover related entities via graph traversal
- `GET /health`: Health check and service status
- `GET /stats`: Processing statistics and metrics

## Technology Stack

### Core Framework
- **FastAPI**: Async web framework for REST API
- **Uvicorn**: ASGI server for production deployment
- **Pydantic**: Data validation and settings management

### ML/AI Components
- **GLiNER**: Zero-shot entity recognition model for hierarchical entity extraction
- **vLLM**: High-performance LLM inference for relationship extraction
- **HTTPX**: Async HTTP client for vLLM communication

### Storage
- **Neo4j**: Graph database for knowledge persistence
- **Neo4j Python Driver**: Official async driver for graph operations

### Data Processing
- **YAML**: Entity taxonomy definition
- **JSON**: LLM response parsing and API communication

## Design Principles

### Separation of Concerns
Each module handles a distinct responsibility:
- API layer manages HTTP communication
- Extractors focus solely on ML model inference
- Pipeline orchestrates workflow without business logic
- Storage abstracts graph database operations

### Async-First Architecture
All I/O-bound operations use async/await:
- Neo4j database operations
- vLLM inference requests
- FastAPI request handling
- Concurrent processing support

### Stateless Processing
Each document processing request is independent:
- No shared state between requests
- Idempotent operations for retry safety
- Deterministic entity normalization

### Schema-Driven Integration
Pydantic models define clear contracts:
- Request validation at API boundary
- Type safety throughout processing pipeline
- Self-documenting API via OpenAPI spec

## Performance Characteristics

### Processing Throughput
- Entity extraction: ~2-3 seconds per document (2000-5000 tokens)
- Relationship extraction: ~5-10 seconds per document
- Total pipeline: ~10-15 seconds per document
- Concurrent processing: Up to 8 documents in parallel

### Storage Efficiency
- Neo4j query response: <100ms for entity search
- Chunk retrieval: <200ms for 100 chunks
- Entity expansion: <150ms for depth-1 traversal

### Resource Requirements
- Memory: ~4GB for GLiNER model
- CPU: Multi-core recommended for concurrent processing
- Neo4j: 2-4GB heap for production workloads

## Operational Considerations

### Error Handling
- Graceful degradation when vLLM unavailable
- Automatic retry with exponential backoff
- Detailed logging at INFO, DEBUG, and ERROR levels
- Health check endpoint for monitoring

### Scalability
- Horizontal scaling via multiple service instances
- Shared Neo4j cluster for distributed storage
- Stateless design enables load balancing

### Monitoring
- Processing statistics tracking
- Per-request timing metrics
- Service health indicators
- Model availability status

## Next Steps

For detailed technical documentation, navigate to specific module pages:

- **[Architecture](architecture.md)**: Understand system design and data flow
- **[API](api.md)**: Learn endpoint specifications and integration patterns
- **[Extractors](extractors.md)**: Deep dive into ML model integration
- **[Pipeline](pipeline.md)**: Explore processing orchestration
- **[Storage](storage.md)**: Master graph database operations
- **[Configuration](configuration.md)**: Configure and deploy the service
