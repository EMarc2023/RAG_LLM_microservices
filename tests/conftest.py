import pytest
from fastapi.testclient import TestClient

# Import your apps using the new package structure
from microservices.llm_app import app as llm_app
from microservices.rag_app import app as rag_app
from microservices.rag_llm_app import app as rag_llm_app


@pytest.fixture
def llm_client():
    return TestClient(llm_app)


@pytest.fixture
def rag_client():
    return TestClient(rag_app)


@pytest.fixture
def combiner_client():
    return TestClient(rag_llm_app)
