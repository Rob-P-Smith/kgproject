"""
KG Service Client for mcpragcrawl4ai

HTTP client to communicate with kg-service from mcpragcrawl4ai.
Place this file in: mcpragcrawl4ai/core/clients/kg_service_client.py
"""

import logging
import httpx
from typing import Dict, List, Optional, Any
import backoff

logger = logging.getLogger(__name__)


class KGServiceError(Exception):
    """Base exception for KG service errors"""
    pass


class KGServiceUnavailable(KGServiceError):
    """Raised when KG service is unavailable"""
    pass


class KGServiceClient:
    """
    HTTP client for kg-service API

    Handles communication between mcpragcrawl4ai and kg-service
    for entity/relationship extraction.
    """

    def __init__(
        self,
        base_url: str = "http://kg-service:8088",
        timeout: float = 300.0,  # 5 minutes
        max_retries: int = 3
    ):
        """
        Initialize KG service client

        Args:
            base_url: kg-service API URL (container name or IP)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5
            )
        )

        logger.info(f"KGServiceClient initialized: {self.base_url}")

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @backoff.on_exception(
        backoff.expo,
        (httpx.RequestError, httpx.HTTPStatusError),
        max_tries=3,
        max_time=60
    )
    async def health_check(self) -> Dict:
        """
        Check if kg-service is healthy

        Returns:
            Health status dict

        Raises:
            KGServiceUnavailable: If service is not responding
        """
        try:
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()

        except httpx.RequestError as e:
            logger.error(f"KG service health check failed: {e}")
            raise KGServiceUnavailable(f"Cannot reach kg-service: {e}")

        except httpx.HTTPStatusError as e:
            logger.error(f"KG service returned error: {e.response.status_code}")
            raise KGServiceUnavailable(
                f"kg-service unhealthy: {e.response.status_code}"
            )

    async def get_stats(self) -> Dict:
        """
        Get kg-service statistics

        Returns:
            Statistics dict
        """
        try:
            response = await self.client.get(f"{self.base_url}/stats")
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.warning(f"Failed to get kg-service stats: {e}")
            return {}

    async def ingest_document(
        self,
        content_id: int,
        url: str,
        title: str,
        markdown: str,
        chunks: List[Dict],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Send document to kg-service for processing

        Args:
            content_id: SQLite content ID
            url: Document URL
            title: Document title
            markdown: Full cleaned markdown
            chunks: List of chunk metadata dicts with:
                - vector_rowid: content_vectors rowid
                - chunk_index: sequential number
                - char_start: position in markdown
                - char_end: end position
                - text: chunk text
            metadata: Additional metadata (tags, timestamp, etc.)

        Returns:
            Dict with processing results:
                - success: bool
                - neo4j_document_id: str
                - entities_extracted: int
                - relationships_extracted: int
                - entities: List[Dict]
                - relationships: List[Dict]
                - summary: Dict

        Raises:
            KGServiceError: If processing fails
        """

        request_data = {
            "content_id": content_id,
            "url": url,
            "title": title,
            "markdown": markdown,
            "chunks": chunks,
            "metadata": metadata or {}
        }

        logger.info(f"Sending document to kg-service: content_id={content_id}")
        logger.debug(f"  URL: {url}")
        logger.debug(f"  Markdown length: {len(markdown)} chars")
        logger.debug(f"  Chunks: {len(chunks)}")

        try:
            response = await self.client.post(
                f"{self.base_url}/api/v1/ingest",
                json=request_data,
                timeout=self.timeout
            )

            response.raise_for_status()
            result = response.json()

            logger.info(f"✓ KG processing complete: content_id={content_id}")
            logger.info(f"  Entities: {result.get('entities_extracted', 0)}")
            logger.info(f"  Relationships: {result.get('relationships_extracted', 0)}")
            logger.info(f"  Time: {result.get('processing_time_ms', 0)}ms")

            return result

        except httpx.TimeoutException:
            error_msg = f"KG processing timeout after {self.timeout}s"
            logger.error(f"✗ {error_msg}")
            raise KGServiceError(error_msg)

        except httpx.HTTPStatusError as e:
            error_msg = f"KG service error {e.response.status_code}: {e.response.text}"
            logger.error(f"✗ {error_msg}")
            raise KGServiceError(error_msg)

        except httpx.RequestError as e:
            error_msg = f"Failed to connect to kg-service: {e}"
            logger.error(f"✗ {error_msg}")
            raise KGServiceUnavailable(error_msg)

        except Exception as e:
            error_msg = f"Unexpected error during KG processing: {e}"
            logger.error(f"✗ {error_msg}", exc_info=True)
            raise KGServiceError(error_msg)

    async def ingest_document_safe(
        self,
        content_id: int,
        url: str,
        title: str,
        markdown: str,
        chunks: List[Dict],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict]:
        """
        Send document to kg-service with error handling

        Same as ingest_document but returns None instead of raising
        exceptions. Use this for non-critical KG processing where
        failures should not block the main pipeline.

        Args:
            Same as ingest_document

        Returns:
            Processing result dict or None if failed
        """
        try:
            return await self.ingest_document(
                content_id=content_id,
                url=url,
                title=title,
                markdown=markdown,
                chunks=chunks,
                metadata=metadata
            )

        except KGServiceUnavailable as e:
            logger.warning(f"KG service unavailable: {e}")
            return None

        except KGServiceError as e:
            logger.error(f"KG processing failed: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error in KG processing: {e}", exc_info=True)
            return None


# ============================================================================
# Usage Example
# ============================================================================

async def example_usage():
    """Example usage of KG service client"""

    async with KGServiceClient("http://kg-service:8088") as client:
        # Health check
        health = await client.health_check()
        print(f"KG Service Status: {health['status']}")

        # Send document for processing
        result = await client.ingest_document(
            content_id=123,
            url="https://docs.fastapi.com",
            title="FastAPI Documentation",
            markdown="# FastAPI\n\nFastAPI is a modern web framework...",
            chunks=[
                {
                    "vector_rowid": 45001,
                    "chunk_index": 0,
                    "char_start": 0,
                    "char_end": 2500,
                    "text": "# FastAPI\n\nFastAPI is..."
                },
                {
                    "vector_rowid": 45002,
                    "chunk_index": 1,
                    "char_start": 2450,
                    "char_end": 4950,
                    "text": "...modern web framework..."
                }
            ],
            metadata={
                "tags": "python,web,api",
                "timestamp": "2025-10-15T12:00:00Z"
            }
        )

        print(f"Entities extracted: {result['entities_extracted']}")
        print(f"Relationships extracted: {result['relationships_extracted']}")
        print(f"Neo4j document ID: {result['neo4j_document_id']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
