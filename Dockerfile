# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Stage 1 — builder: install dependencies with uv into /app/.venv
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Install uv from the official distro image layer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first for layer-cache friendliness
COPY pyproject.toml uv.lock ./

# Install only runtime deps (base + [real] extras — pgvector, neo4j, langfuse)
# into an in-tree venv; skip dev tools.
# --frozen: respect the exact lockfile; --no-dev: no pytest/mypy/ruff.
# The [real] extras group carries langfuse, neo4j, and ragas — all optional
# at runtime but baked in so the image works out-of-the-box.
# Fallback to base-only sync if [real] extra deps fail (e.g., platform gap).
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
RUN uv sync --frozen --extra real --no-dev 2>/dev/null || uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Stage 2 — runtime: lean final image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Install netcat for the HEALTHCHECK (TCP probe — no HTTP /healthz endpoint);
# purge apt lists to trim image size.
RUN apt-get update && apt-get install -y --no-install-recommends netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (uid 1001)
RUN useradd --system --create-home --uid 1001 --gid 0 appuser

WORKDIR /app

# Copy the pre-built venv from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY src/ /app/src/

# Put the venv's bin on PATH so console-scripts resolve without activation
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Switch to non-root before the HEALTHCHECK / CMD
USER appuser

EXPOSE 8000

# TCP-only healthcheck — atlas-mcp has no /healthz HTTP endpoint.
# nc exits 0 when the port is open, 1 when connection is refused.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD nc -z localhost 8000 || exit 1

CMD ["atlas-mcp", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
