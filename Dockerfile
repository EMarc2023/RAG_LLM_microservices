FROM python:3.10-slim

WORKDIR /app

# 1. Install system-level dependencies
# libgomp1 is critical for FAISS on Linux
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# 2. Install CPU-specific Torch first to keep image size down
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the application code
COPY . .

EXPOSE 8000 8001 8002

# We define the actual startup command in docker-compose.yml