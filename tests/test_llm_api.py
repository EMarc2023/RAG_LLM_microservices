import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import MagicMock
from microservices.llm_app import app, get_llm, breaker

# 1. SETUP: Dependency Override
# This is the "Enterprise" way to test. We replace the real model with a Mock.
mock_model = MagicMock()


async def override_get_llm():
    return mock_model


app.dependency_overrides[get_llm] = override_get_llm


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def async_client():
    # Using 'async with' triggers the lifespan (startup/shutdown)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# 2. THE TESTS


@pytest.mark.anyio
async def test_generate_success(async_client):
    # Arrange
    mock_prompt = "What is the capital of France?"
    # Your splitter logic looks for <|assistant|> and takes what's after it
    # flake8: noqa: E501
    mock_model.return_value = [{"generated_text": f"{mock_prompt}<|assistant|>\nParis"}]
    await breaker.record_success()

    # Act
    response = await async_client.post("/generate", params={"prompt": "Hello"})

    # Assert
    assert response.status_code == 200
    # CHANGE THIS LINE:
    assert response.json()["response"] == "Paris"


@pytest.mark.anyio
async def test_circuit_breaker_open(async_client):
    # Arrange: Force the circuit to open
    breaker.is_open = True

    # Act
    response = await async_client.post("/generate", params={"prompt": "Hello"})

    # Assert
    assert response.status_code == 503
    assert "Circuit open" in response.json()["detail"]

    # Clean up for next tests
    await breaker.record_success()


@pytest.mark.anyio
async def test_inference_failure_opens_circuit(async_client):
    # Arrange: Mock a crash
    mock_model.side_effect = Exception("GPU/CPU Crash")

    # Act: Trigger failures up to the threshold (e.g., 5)
    for _ in range(5):
        await async_client.post("/generate", params={"prompt": "Fail me"})

    # Assert
    assert breaker.is_open is True

    # Clean up
    await breaker.record_success()
    mock_model.side_effect = None
