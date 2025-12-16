import pytest
import respx
import httpx
from httpx import Response
from rag_llm_app import ask_ai, clients, settings
from fastapi import HTTPException


@pytest.mark.asyncio
async def test_ask_ai_happy_path_simple():
    # 1. Start the mock
    with respx.mock(assert_all_called=False) as respx_mock:
        # Mock the RAG service
        respx_mock.get(url__startswith=f"{settings.rag_service_url}/ask").mock(
            return_value=Response(
                200, json={"answer": "The capital of France is Paris."}
            )
        )
        # Mock the LLM service
        # flake8: noqa: E501
        respx_mock.post(url__startswith=f"{settings.llm_service_url}/generate").mock(
            return_value=Response(
                200, json={"response": "Paris is the capital of France."}
            )
        )

        # 2. Setup a real AsyncClient and inject it directly
        async with httpx.AsyncClient() as ac:
            clients.http_client = ac

            # 3. CALL THE FUNCTION DIRECTLY (No test_client needed)
            response = await ask_ai(query="What is the capital?")

        # 4. Assertions on the dictionary returned by the function
        assert response["query"] == "What is the capital?"
        assert "Paris" in response["ai_answer"]
        assert "France" in response["context_used"]


@pytest.mark.asyncio
async def test_ask_ai_timeout_simple():
    with respx.mock(assert_all_called=False) as respx_mock:
        respx_mock.get(url__startswith=f"{settings.rag_service_url}/ask").mock(
            return_value=Response(200, json={"answer": "Some context."})
        )
        # flake8: noqa: E501
        respx_mock.post(url__startswith=f"{settings.llm_service_url}/generate").mock(
            side_effect=httpx.TimeoutException("Mocked Timeout")
        )

        async with httpx.AsyncClient() as ac:
            clients.http_client = ac

            # Catch the FastAPI HTTPException your code re-raises
            with pytest.raises(HTTPException) as excinfo:
                await ask_ai(query="Slow prompt")

            assert excinfo.value.status_code == 500
            assert "Internal Orchestrator Error" in excinfo.value.detail
