# Dockerfile — Content Moderation RL Environment
# Builds a lightweight Python 3.11 image that serves the OpenEnv HTTP API.
#
# Build:  docker build -t cmp-env .
# Run:    docker run -p 7860:7860 cmp-env
# Test:   curl http://localhost:7860/health

FROM python:3.11-slim

# Metadata
LABEL maintainer="CMP Team"
LABEL description="Content Moderation RL Environment — OpenEnv HTTP Server"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install Python deps first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose the port used by HuggingFace Spaces and the validator
EXPOSE 7860

# Environment variables (can be overridden at runtime)
ENV DATASET_SEED=42
ENV DATASET_SIZE=500
ENV PYTHONUNBUFFERED=1

# Start the FastAPI server via uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
