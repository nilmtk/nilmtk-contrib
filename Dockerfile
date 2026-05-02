# Use slim Python 3.11 image
FROM python:3.11.5-slim as base

# Avoid prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libatlas-base-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv first
RUN pip install --no-cache-dir uv

# Copy project files (assumes everything is in one dir)
COPY . .

# Install the package with all optional backends. Use a narrower extra such as
# .[torch], .[tensorflow], or .[classical] for backend-specific production images.
RUN uv pip install --system ".[all]"


# Optional: install dev dependencies too
# RUN uv pip install --system ".[dev]"

# Set env vars
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command (can be overridden)
CMD [ "python" ]
