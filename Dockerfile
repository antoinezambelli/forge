FROM python:3.13-alpine3.23 AS builder

RUN pip install --root-user-action=ignore --no-cache-dir --upgrade pip \
    && pip install --root-user-action=ignore --no-cache-dir uv

ENV UV_LINK_MODE=copy

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --extra anthropic --no-install-project --no-dev

COPY pyproject.toml LICENSE README.md ./
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra anthropic --no-dev


FROM python:3.13-alpine3.23
LABEL org.opencontainers.image.title="Forge Proxy" \
    org.opencontainers.image.description="A reliability layer for self-hosted LLM tool-calling" \
    org.opencontainers.image.url="https://github.com/antoinezambelli/forge" \
    org.opencontainers.image.source="https://github.com/antoinezambelli/forge" \
    org.opencontainers.image.vendor="Antoine Zambelli" \
    org.opencontainers.image.licenses="MIT"
ENV PYTHONUNBUFFERED=1

RUN apk add --no-cache ca-certificates \
    && addgroup -g 1000 appuser \
    && adduser -D -u 1000 -G appuser appuser

COPY --from=builder --chown=appuser:appuser /app /app

WORKDIR /app

USER appuser

ENV PATH="/app/.venv/bin:$PATH"

HEALTHCHECK \
    --interval=15s \
    --timeout=5s \
    --start-period=5s \
    --retries=3 \
    CMD ["wget", "--no-verbose", "--tries=1", "--spider", "http://127.0.0.1:8081/health"]

EXPOSE 8081

ENTRYPOINT ["forge-proxy", "--host", "0.0.0.0", "--port", "8081"]
