import httpx
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from typing import Optional

# Pydantic for settings
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger()

# Config - In a real app, move these to your .env
RAG_SERVICE_URL = "http://127.0.0.1:8000"  # Port where your RAG app runs
LLM_SERVICE_URL = "http://127.0.0.1:8001"  # Port where your LLM app runs


# Define the class (It automatically handles loading from .env)
class Settings(BaseSettings):

    rag_service_url: str = "http://127.0.0.1:8000"
    llm_service_url: str = "http://127.0.0.1:8001"
    api_key: str
    otel_service_name: str = "Mini-RAG-API"

    # NEW: Nested configuration for Pydantic Settings
    model_config = SettingsConfigDict(
        # This tells Pydantic to load variables from a file named '.env'
        env_file=".env",
        # This tells Pydantic to look for .env in the current directory
        env_file_encoding="utf-8",
    )


# Instantiate the settings object globally
settings = Settings()


# 1. SHARED STATE CONTAINER
class ServiceClients:
    http_client: httpx.AsyncClient = None


clients = ServiceClients()


# 2. LIFESPAN (The "Production" way)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Initialize a single, persistent client for the whole app
    # We set limits here to handle high concurrency
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    clients.http_client = httpx.AsyncClient(limits=limits, timeout=30.0)
    logger.info("orchestrator_startup", status="http_client_ready")

    yield

    # SHUTDOWN: Gracefully close all connections
    await clients.http_client.aclose()
    logger.info("orchestrator_shutdown")


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


@app.post("/ask_ai")
# flake8: noqa: E501
async def ask_ai(request: Optional[QueryRequest] = None, query: Optional[str] = None):
    # 1. FIX THE 422: Extract query from Body or URL Param
    final_query = query or (request.query if request else None)

    if not final_query:
        raise HTTPException(
            status_code=422,
            detail="Query missing. Provide it in JSON body or as ?query=...",
        )

    # Use the persistent client from ServiceClients for better performance
    client = clients.http_client

    try:
        # STEP 1: RETRIEVE (GET request to RAG)
        logger.info("orchestrator_retrieval_start", query=final_query)
        rag_response = await client.get(
            f"{settings.rag_service_url}/ask",
            params={"query": final_query},
            headers={"X-API-Key": settings.api_key},
        )
        rag_response.raise_for_status()

        # Get context from RAG's "answer" field
        context = rag_response.json().get("answer", "No context found.")

        # STEP 2: AUGMENT
        # flake8: noqa: E501
        prompt = (
            f"You are a helpful assistant. Use the following context to answer the question.\n"
            f"Context: {context}\n"
            f"Question: {final_query}\n"
            f"Answer:"
        )

        # STEP 3: GENERATE (POST request to LLM)
        logger.info("orchestrator_generation_start")

        # flake8: noqa: E501
        # Change 'json=' to 'params=' to send it in the URL as the LLM service expects
        llm_response = await client.post(
            f"{settings.llm_service_url}/generate",
            params={"prompt": prompt},  # <--- Use params here
        )
        llm_response.raise_for_status()
        # flake8: noqa: E501
        final_answer = llm_response.json().get("response", "No response from LLM.")

        # Final results
        logger.info("orchestrator_success", query=final_query)
        return {
            "query": final_query,
            "context_used": context,
            "ai_answer": final_answer,
        }

    except httpx.HTTPStatusError as e:
        logger.error(
            "service_call_failed",
            status_code=e.response.status_code,
            detail=e.response.text,
        )
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Downstream service error: {e.response.text}",
        )
    except Exception as e:
        logger.error("orchestrator_error", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Internal Orchestrator Error: {str(e)}"
        )
