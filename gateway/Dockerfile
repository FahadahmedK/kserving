FROM python:3.12-slim

ARG PROJECT_DIR

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY ${PROJECT_DIR}/uv.lock /app/uv.lock
COPY ${PROJECT_DIR}/pyproject.toml /app/pyproject.toml

# Copy the frontend source code
COPY ${PROJECT_DIR}/gateway /app/gateway

# copy local dependency code
# COPY ${PROJECT_DIR}/model /app/model
ENV UV_HTTP_TIMEOUT=300
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv uv sync --package="gateway" --frozen --no-install-project

RUN uv sync --package="gateway" --frozen

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 5000

CMD ["uvicorn", "gateway.app:app","--host=0.0.0.0", "--port=5000"]