# 📖 Mini RAG Pipeline: Production-Grade Retrieval-Augmented Generation (RAG) API

[![Python CI/CD](https://github.com/EMarc2023/Mini_RAG_pipeline/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/EMarc2023/Mini_RAG_pipeline/actions/workflows/ci_cd.yml)

## 🚀 Overview

This repository hosts a production-ready **Retrieval-Augmented Generation (RAG)** pipeline implemented as a RESTful API using **FastAPI**. It is designed for high performance, reliability, and enterprise scalability, leveraging established Python libraries like PyTorch (CPU-only), HuggingFace Transformers, and LangChain components.

The core function is to allow users to submit queries against a pre-loaded knowledge base and retrieve contextually grounded answers, ensuring minimal hallucinations and providing verifiable sources.

## ✨ Production-Grade Features

This codebase was developed with several key production-grade features and engineering considerations in mind:

| Feature | Implementation | Benefit |
| :--- | :--- | :--- |
| **Containerization** | Includes a robust `Dockerfile` and a CI/CD process to build and generate a portable Docker image artifact. | Guarantees environmental consistency across development, testing, and production (Dev/Test Parity). |
| **API Key Security** | Implements a custom middleware/dependency to enforce authentication via an `X-API-Key` header. | Prevents unauthorized access and protects the underlying LLM/RAG resources. |
| **Dependency Control** | Uses a dedicated `requirements.txt` and a strategic, explicit installation of **CPU-only PyTorch**. | Minimizes image size (preventing CI disk space exhaustion) and ensures compatibility with non-GPU cloud/local environments. |
| **CI/CD Validation** | GitHub Actions workflow (`ci_cd.yml`) enforces linting (`black`, `flake8`), unit testing (`pytest`), and successfully generates the deployment artifact. | Ensures code quality, functionality, and automated artifact generation on every push to `main`. |
| **Configuration** | Uses a `.env` file (or environment variables) for sensitive keys and configuration parameters (e.g., `API_KEY`, `OTEL_SERVICE_NAME`). | Decouples configuration from code for security and environment flexibility. |

### Key Files

* **`app.py`**: Initializes the FastAPI application, loads the RAG pipeline (vector store, embeddings, LLM), and defines the secured `/ask` endpoint.
* **`Dockerfile`**: Based on a slim Python image, copies dependencies, installs packages (with the CPU-only PyTorch index), and sets the startup command.
* **`ci_cd.yml`**: Includes critical steps like running unit tests, linting checks, and an **aggressive disk cleanup step** necessary for successfully building and saving large ML-based Docker images on GitHub Actions runners.

## ⚙️ Setup and Local Run (WSL/Linux)

### Prerequisites

* Python 3.10+
* Docker Desktop (if running containerized)
* `git`

### Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone [Your-Repo-URL]
    cd mini-rag-pipeline
    ```
2.  **Create and fill `.env`:** Copy `.env.example` to `.env` and fill in your actual `API_KEY` and other necessary environment variables.
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the API Locally (WSL/Linux)

Run the API using Uvicorn, passing the environment variables:

```bash
# Set your actual API Key and run the server
API_KEY="SECRET_TEST_KEY_123" uvicorn app:app --host 0.0.0.0 --port 8000
