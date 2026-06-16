# NILMBench2026 / nilmtk-contrib container image.
#
# Build the default all-backend image:
#   docker build -t nilmtk-contrib:all .
#
# Build a narrower backend image:
#   docker build -t nilmtk-contrib:torch --build-arg INSTALL_EXTRA=torch .
#   docker build -t nilmtk-contrib:tensorflow --build-arg INSTALL_EXTRA=tensorflow .
#   docker build -t nilmtk-contrib:classical --build-arg INSTALL_EXTRA=classical .
#
# Include dev/test dependencies:
#   docker build -t nilmtk-contrib:dev --build-arg INSTALL_DEV=true .

FROM python:3.11-slim-bookworm

ARG INSTALL_EXTRA=all
ARG INSTALL_DEV=false

LABEL org.opencontainers.image.title="nilmtk-contrib"
LABEL org.opencontainers.image.description="NILMBench2026 benchmark toolkit with NILMTK-compatible disaggregation models"
LABEL org.opencontainers.image.source="https://github.com/sustainability-lab/nilmbench"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

# System libraries for NumPy/SciPy/scikit-learn wheels and compiling native extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    gfortran \
    git \
    libffi-dev \
    libgomp1 \
    libopenblas-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir "uv>=0.5"

# Copy only install inputs first for better layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY nilmtk_contrib/ nilmtk_contrib/

RUN uv pip install --system ".[${INSTALL_EXTRA}]" \
    && if [ "${INSTALL_DEV}" = "true" ]; then uv pip install --system ".[dev]"; fi

# Optional runtime assets (tests, notebooks) live outside the install layer.
COPY tests/ tests/
COPY sample_notebooks/ sample_notebooks/

CMD ["bash"]
