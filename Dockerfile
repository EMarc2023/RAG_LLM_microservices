# Dockerfile
# Base stage: Use a lightweight Python base image
FROM python:3.10-slim

# Set environment variables for non-interactive commands
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
# This caches the dependency layer if requirements.txt hasn't changed
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose the port FastAPI runs on
EXPOSE 8000

# Set the entrypoint to run the server
# The CMD provides the default arguments for Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
