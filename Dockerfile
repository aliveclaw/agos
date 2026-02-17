FROM python:3.13-slim

LABEL maintainer="aliveclaw" \
      description="AGenticOS — Your Agentic Operating System" \
      version="0.1.0"

# Prevent Python from writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps + runtimes for agent workloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential && \
    # Node.js 20 LTS (for OpenClaw)
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    # Go 1.22 (for PicoClaw)
    curl -fsSL https://go.dev/dl/go1.22.10.linux-amd64.tar.gz | tar -C /usr/local -xz && \
    # Cleanup
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/go/bin:$PATH"

# Install dependencies first (layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e "." 2>/dev/null || true

# Copy full source (includes workloads/)
COPY . .

# Install the package
RUN pip install --no-cache-dir -e "."

# Initialize workspace
RUN python -m agos init 2>/dev/null || true

# Copy agent workloads to /workloads/ (OS discovers them at boot)
RUN mkdir -p /workloads && cp -r /app/workloads/* /workloads/

# Build PicoClaw (Go agent)
RUN cd /workloads/picoclaw && go build -o picoclaw . || true

# Dashboard port
EXPOSE 8420

# Launch the OS — auto-discovers and supervises agent workloads
CMD ["python", "-m", "agos.serve"]
