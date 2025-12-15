# Imports at top (conventional)
import json
import asyncio
import logging
from typing import List
from fastapi import FastAPI, HTTPException, Depends, Header, status
from contextlib import asynccontextmanager  # ADD THIS

# Pydantic for settings
from pydantic_settings import BaseSettings, SettingsConfigDict

# Telemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

# flake8: noqa: E501
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)  # Use Batch for performance

# Small CPU-friendly LLM + embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


# Define the class (It automatically handles loading from .env)
class Settings(BaseSettings):

    # NEW: Nested configuration for Pydantic Settings
    model_config = SettingsConfigDict(
        # This tells Pydantic to load variables from a file named '.env'
        env_file=".env",
        # This tells Pydantic to look for .env in the current directory
        env_file_encoding="utf-8",
    )

    api_key: str
    otel_service_name: str = "Mini-RAG-API"


# Instantiate the settings object globally
settings = Settings()

# ------------------------
# Telemetry setup (REVISED FOR JAEGER)
# ------------------------

# 1. Define Resource (Service Name)
resource = Resource.create({"service.name": "Mini-RAG-API"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

# 2. Configure Jaeger Exporter
# This targets the Jaeger gRPC Collector on the standard port 14250
otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",  # Standard OTLP gRPC port
    insecure=True,  # OTLP uses a secure connection, but not local Jaeger
)

# 3. Configure Tracer Provider
trace_provider = TracerProvider(
    resource=Resource.create({"service.name": settings.otel_service_name})
)
trace_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(trace_provider)

# ------------------------
# Logging
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------
# Data pipeline
# ------------------------
class RAGPipeline:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = list(self._load_documents())
        self.index = self._build_index()

    # Iterator / generator to load documents
    def _load_documents(self):
        with open(self.dataset_path, "r") as f:
            for line in f:
                yield json.loads(line)["text"]

    # Build FAISS vector index
    def _build_index(self):
        embeddings = []
        for doc_batch in self._batch_iterator(self.documents, batch_size=2):
            emb = self.embedding_model.encode(doc_batch, convert_to_numpy=True)
            embeddings.extend(emb)
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))
        return index

    # Iterator for batching
    def _batch_iterator(self, items: List[str], batch_size: int):
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    # Async generator for retrieval
    async def retrieve_generator(self, query: str, top_k: int = 2):
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True)
        _, indx = self.index.search(query_emb, top_k)
        for idx in indx[0]:
            await asyncio.sleep(0.05)  # simulate I/O
            yield self.documents[idx]

    # Async final answer (simulated LLM)
    async def answer(self, query: str):
        retrieved_docs = []
        async for doc in self.retrieve_generator(query):
            retrieved_docs.append(doc)
        # Simulate LLM combining retrieved docs
        return " | ".join(retrieved_docs)


# Context manager for RAG resources
rag_resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # **Startup Event** (Runs once before the server accepts connections)
    print("--- Starting RAG Pipeline Initialization ---")

    # 1. Initialize the heavy resource (RAGPipeline)
    # rag_resources["pipeline"] = RAGPipeline()
    rag_resources["pipeline"] = RAGPipeline(dataset_path="sample_data.jsonl")

    print("--- RAG Pipeline Ready ---")
    yield  # The server starts accepting requests here

    # **Shutdown Event** (Runs once when the server is shutting down)
    print("--- Shutting Down RAG Pipeline ---")
    # 2. Cleanup (e.g., closing database connections, which we don't have here)
    rag_resources.clear()
    print("--- Shutdown Complete ---")


# ------------------------
# FastAPI setup
# ------------------------
app = FastAPI(lifespan=lifespan)

# ------------------------
# FIX: Change Global Instantiation to Lazy Getter
# ------------------------

# 1. Global variable to store the pipeline once it's built
_pipeline_instance: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    """
    Lazy initializes and returns the RAGPipeline instance.
    This ensures the pipeline is only built ONCE and only when first needed.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        logger.info("Initializing RAGPipeline...")
        # This instantiation only happens after the tests have written data
        _pipeline_instance = RAGPipeline("sample_data.jsonl")
        logger.info("RAGPipeline initialized successfully.")
    return _pipeline_instance


def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """Verifies a simple API key present in the request header."""
    # CRITICAL: Change from hardcoded key to settings.api_key
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    # Return the "user" identity if successful
    return {"user_id": 1, "role": "admin"}


# ------------------------
# Async endpoint with telemetry
# ------------------------
@app.get("/")
async def home():
    return {"message": "Welcome to the Mini RAG-LLM API"}


@app.get("/ask")
async def ask_question(query: str, _: dict = Depends(verify_api_key)):

    try:
        pipeline = rag_resources.get("pipeline")

        if pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="RAG Pipeline not initialized.",
            )

        logger.info(f"Received query: {query}")

        with tracer.start_as_current_span("ask_question_endpoint") as span:
            answer = await pipeline.answer(query)
            span.set_attribute("query_length", len(query))
            span.set_attribute("answer_length", len(answer))
        logger.info(f"Answer generated: {answer}")
        return {"query": query, "answer": answer}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
