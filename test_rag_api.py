import pytest
import pytest_asyncio  # ADD THIS IMPORT
import asyncio
import json
from fastapi import HTTPException
from httpx import AsyncClient, ASGITransport
from rag_app import app, RAGPipeline, verify_api_key, rag_resources

SAMPLE_FILE = "sample_data.jsonl"
SAMPLE_DOCS = [
    # flake8: noqa: E501
    {
        "text": "FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python typehints."
    },
    # flake8: noqa: E501
    {
        "text": "FAISS is a library for efficient similarity search and clustering of dense vectors."
    },
    # flake8: noqa: E501
    {
        "text": "Embedding models turn text into numerical vectors that capture semantic meaning."
    },
]

# Write the sample data to JSONL
with open(SAMPLE_FILE, "w") as f:
    for doc in SAMPLE_DOCS:
        f.write(json.dumps(doc) + "\n")


@pytest_asyncio.fixture
async def initialize_app_lifespan():
    """Manually triggers the lifespan startup event for testing."""

    # 1. Manually run the startup logic from app.py
    # We copy the exact logic from the 'lifespan' function's startup section

    print("--- Running manual RAG Pipeline Initialization for tests ---")
    rag_resources["pipeline"] = RAGPipeline(
        "sample_data.jsonl"
    )  # Pass the required dataset path
    print("--- RAG Pipeline Ready for tests ---")

    yield  # Run the tests

    # 2. Manually run the shutdown logic
    print("--- Shutting Down RAG Pipeline after tests ---")
    rag_resources.clear()
    print("--- Shutdown Complete ---")


# ------------------------
# Class-level tests
# ------------------------
@pytest.mark.asyncio
async def test_ragpipeline_basic():
    pipeline = RAGPipeline(SAMPLE_FILE)

    # Test documents loaded correctly
    docs = list(pipeline._load_documents())
    assert len(docs) == len(SAMPLE_DOCS)
    assert docs[0] == SAMPLE_DOCS[0]["text"]

    # Test batch iterator
    batches = list(pipeline._batch_iterator(docs, batch_size=2))
    assert len(batches) == 2
    assert len(batches[0]) == 2

    # Test async generator retrieval
    retrieved = []
    async for doc in pipeline.retrieve_generator("FastAPI", top_k=2):
        retrieved.append(doc)
    assert len(retrieved) <= 2
    assert any("FastAPI" in d for d in retrieved)

    # Test answer function
    answer_text = await pipeline.answer("FastAPI")
    assert "FastAPI" in answer_text or len(answer_text) > 0


# 1. Define the mock function that always passes
def mock_auth_success():
    """A mock that always succeeds and returns a test user."""
    return {"user_id": 999, "role": "test_runner"}


# 2. Define the mock function that always fails
def mock_auth_failure():
    """A mock that always raises an error."""
    raise HTTPException(status_code=401, detail="Mocked Auth Failed")


@pytest.mark.asyncio
async def test_ask_endpoint():
    app.dependency_overrides[verify_api_key] = mock_auth_success
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as ac:
            # Test normal query
            response = await ac.get("/ask", params={"query": "FastAPI"})
            assert response.status_code == 200
            json_data = response.json()
            assert "query" in json_data
            assert "answer" in json_data
            assert "FastAPI" in json_data["answer"]
            assert len(json_data["answer"]) > 0

            # Test empty query
            response_empty = await ac.get("/ask", params={"query": ""})
            assert response_empty.status_code == 200
            json_empty = response_empty.json()
            assert "answer" in json_empty

            # Test multiple queries sequentially
            queries = ["FAISS", "Embeddings", "Async generator"]
            for q in queries:
                res = await ac.get("/ask", params={"query": q})
                assert res.status_code == 200
                assert len(res.json()["answer"]) > 0
    finally:
        app.dependency_overrides = {}

        # ------------------------


# Optional: test multiple async calls concurrently
# ------------------------
@pytest.mark.asyncio
async def test_concurrent_queries():
    app.dependency_overrides[verify_api_key] = mock_auth_success
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as ac:
            queries = ["FastAPI", "FAISS", "Embedding"]
            tasks = [ac.get("/ask", params={"query": q}) for q in queries]
            responses = await asyncio.gather(*tasks)
            for res in responses:
                assert res.status_code == 200
                assert "answer" in res.json()
    finally:
        app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_ask_endpoint_with_auth_success():
    # 3a. Override the real dependency with the mock function
    app.dependency_overrides[verify_api_key] = mock_auth_success

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        # Test normal query (no need to pass a real key, the mock handles it)
        response = await ac.get("/ask", params={"query": "FastAPI"})
        assert response.status_code == 200  # Should succeed
        # You can even check the user data if the endpoint exposed it

    # 3b. IMPORTANT: Clear the override after the test
    app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_ask_endpoint_with_auth_failure():
    # Override with the function that always fails
    app.dependency_overrides[verify_api_key] = mock_auth_failure

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        # Request should fail
        response = await ac.get("/ask", params={"query": "FastAPI"})
        assert response.status_code == 401  # Should fail

    app.dependency_overrides = {}
