FROM python:3.12-slim

ARG PROJECT_DIR

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY ${PROJECT_DIR}/uv.lock /app/uv.lock
COPY ${PROJECT_DIR}/pyproject.toml /app/pyproject.toml

# Copy the frontend source code
COPY ${PROJECT_DIR}/frontend /app/frontend

ENV UV_HTTP_TIMEOUT=300
ENV UV_LINK_MODE=copy

RUN uv sync --package="frontend" --frozen

EXPOSE 8501

CMD ["uv", "run", "--frozen", "streamlit",  "run", "frontend/src/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]