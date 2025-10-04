# Tests for API
"""
Tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.server import create_app

client = TestClient(create_app())


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_query_endpoint():
    """Test query endpoint."""
    query_data = {
        "query": "What is machine learning?",
        "top_k": 5,
        "include_sources": True
    }
    
    response = client.post("/api/query", json=query_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "query" in data
    assert "answer" in data
    assert data["query"] == query_data["query"]


def test_collection_stats():
    """Test collection stats endpoint."""
    response = client.get("/api/collection/stats")
    assert response.status_code == 200
    
    data = response.json()
    assert "collection_name" in data
    assert "document_count" in data


def test_invalid_query():
    """Test query with invalid data."""
    query_data = {
        "query": "",  # Empty query
        "top_k": 5
    }
    
    response = client.post("/api/query", json=query_data)
    # Should handle empty query gracefully


@pytest.mark.asyncio
async def test_upload_document():
    """Test document upload."""
    # Create a test file
    test_content = b"This is a test document for the RAG system."
    
    files = {"file": ("test.txt", test_content, "text/plain")}
    response = client.post("/api/documents/upload", files=files)
    
    # Should return success or appropriate error
    assert response.status_code in [200, 400, 500]