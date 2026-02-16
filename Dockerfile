FROM python:3.13-slim

LABEL maintainer="aliveclaw" \
      description="AGenticOS — Your Agentic Operating System" \
      version="0.1.0"

# Prevent Python from writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e "." 2>/dev/null || true

# Copy full source
COPY . .

# Install the package
RUN pip install --no-cache-dir -e "."

# Initialize workspace
RUN python -m agos init 2>/dev/null || true

# Dashboard port
EXPOSE 8420

# Launch dashboard + live demo engine together
CMD ["python", "-m", "agos.serve"]
