# Dockerfile for kg-service
# Knowledge Graph service for entity and relationship extraction

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download GLiNER model on build (optional - can also download on first run)
# RUN python -c "from gliner import GLiNER; GLiNER.from_pretrained('urchade/gliner_large-v2.1')"

# Copy application code
COPY . .

# Create models directory for caching
RUN mkdir -p /app/models

# Expose API port
EXPOSE 8088

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8088/health || exit 1

# Run the application
CMD ["python", "main.py"]
