"""
Test API endpoints

Basic tests for kg-service API
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client"""
    from api.server import app
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data
    assert data["status"] == "running"


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data
    assert "uptime_seconds" in data


def test_stats_endpoint(client):
    """Test stats endpoint"""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents_processed" in data
    assert "total_entities_extracted" in data
    assert "total_relationships_extracted" in data


def test_model_info_endpoint(client):
    """Test model info endpoint"""
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "gliner" in data
    assert "vllm" in data


def test_ingest_validation_missing_fields(client):
    """Test ingest endpoint validates required fields"""
    response = client.post("/api/v1/ingest", json={})
    assert response.status_code == 422  # Validation error


def test_ingest_validation_invalid_chunks(client):
    """Test ingest endpoint validates chunk ordering"""
    response = client.post("/api/v1/ingest", json={
        "content_id": 123,
        "url": "https://example.com",
        "title": "Test",
        "markdown": "Test content",
        "chunks": [
            {
                "vector_rowid": 1,
                "chunk_index": 1,  # Wrong order
                "char_start": 0,
                "char_end": 10,
                "text": "Test"
            },
            {
                "vector_rowid": 2,
                "chunk_index": 0,  # Should be after 1
                "char_start": 10,
                "char_end": 20,
                "text": "Content"
            }
        ]
    })
    assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
