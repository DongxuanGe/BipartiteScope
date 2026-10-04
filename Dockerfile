FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --system bipartite-scope \
    && useradd --system --gid bipartite-scope --home-dir /var/lib/bipartite-scope bipartite-scope

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY alembic.ini ./alembic.ini
COPY migrations ./migrations
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.0+cpu' \
    && python -m pip install '.[service]'

RUN mkdir -p /var/lib/bipartite-scope/workspaces \
    && chown -R bipartite-scope:bipartite-scope /var/lib/bipartite-scope

USER bipartite-scope

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v3/health/live', timeout=3)"]

CMD ["bipartite-scope", "serve"]
