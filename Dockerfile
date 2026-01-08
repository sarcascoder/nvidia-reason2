# Docker support for Cosmos-Reason2-8B

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make scripts executable
RUN chmod +x *.sh

# Expose Gradio port
EXPOSE 7860

# Set default environment variables
ENV MODEL_PATH=/app
ENV HF_HOME=/app/.cache/huggingface

# Default command - start the UI
CMD ["python3", "app.py", "--host", "0.0.0.0", "--port", "7860"]
