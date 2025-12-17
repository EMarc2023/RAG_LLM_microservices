import asyncio
import time
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from transformers import pipeline
from opentelemetry import trace

# Global Constants
MAX_LENGTH = 50

# --- 1. SETUP ---
logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)


# --- 2. THREAD-SAFE CIRCUIT BREAKER ---
class ThreadSafeCircuitBreaker:
    def __init__(self, threshold=5):
        self.failure_count = 0
        self.threshold = threshold
        self.is_open = False
        self._lock = asyncio.Lock()  # Ensures state changes are atomic

    async def record_failure(self):
        async with self._lock:
            self.failure_count += 1
            if self.failure_count >= self.threshold:
                self.is_open = True
                # flake8: noqa: E501
                logger.error("circuit_breaker_opened", failures=self.failure_count)

    async def record_success(self):
        async with self._lock:
            self.failure_count = 0
            self.is_open = False


# --- 3. LIFESPAN & DEPENDENCY INJECTION ---
# We store the model in a shared state container
class AIModelContainer:
    model: pipeline = None


breaker = ThreadSafeCircuitBreaker()
container = AIModelContainer()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # STARTUP: Load model once
    logger.info("service_startup", action="loading_model")
    # flake8: noqa: E501
    container.model = pipeline("text2text-generation", model="google/flan-t5-small")
    yield
    # SHUTDOWN: Cleanup
    logger.info("service_shutdown")
    container.model = None


app = FastAPI(lifespan=lifespan)


# DI Function: This is how routes get access to the model
def get_llm():
    if container.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return container.model


# --- 4. THE ROUTE ---
@app.post("/generate")
async def generate(prompt: str, model=Depends(get_llm)):
    # Check circuit status (Read is usually safe, but we respect the state)
    if breaker.is_open:
        # flake8: noqa: E501
        raise HTTPException(status_code=503, detail="Circuit open: System recovering")

    # flake8: noqa: E501
    with tracer.start_as_current_span("inference_request") as span:  # Now we use it!
        start_time = time.time()
        try:
            # Add metadata to the trace
            span.set_attribute("ai.prompt_length", len(prompt))

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: model(prompt, max_length=MAX_LENGTH)
            )

            latency = time.time() - start_time

            # Add more data before finishing
            span.set_attribute("ai.latency", latency)
            await breaker.record_success()

            # flake8: noqa: E501
            logger.info("inference_completed", latency=latency, prompt_size=len(prompt))

            return {"response": result[0]["generated_text"]}

        except Exception as e:
            span.record_exception(e)  # Tag the trace with the error
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            await breaker.record_failure()
            logger.error("inference_error", error=str(e))
            raise HTTPException(status_code=500, detail="Inference failed")
