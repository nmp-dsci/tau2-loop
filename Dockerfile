# The demo image: the public, read-only deployment.
#
# One container, no database, no secrets. Everything a visitor browses is baked
# in at build time — the splits and task extracts, the agent folders, the run
# folders, the ledgers and registries, the MLflow snapshot — so what a given
# image serves is exactly what its commit says it serves.
#
# DEMO_MODE is set here rather than in infrastructure, deliberately: the image
# itself is the thing that cannot make model calls, so no Terraform mistake or
# console edit can turn a public URL into a billable one. There is no API key
# and no CLI login in the environment, and the API has no route that reaches a
# model. The tau2 submodule is not in the image: every route reads our own
# committed files.
#
#   docker build -t tau2loop-demo .
#   docker run --rm -p 8081:8080 tau2loop-demo    # no env file, no keys

# ── Stage 1: frontend bundle ────────────────────────────────────────────
FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: runtime ────────────────────────────────────────────────────
FROM python:3.12-slim
WORKDIR /app

# The viewer's runtime deps only: no tau2, no litellm, no mlflow, no SDK.
RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.32" "pydantic>=2.9" "pyyaml>=6.0" "python-dotenv>=1.0" "rich>=13.9"

# Code, UI, and the committed artifacts the read-only routes serve.
COPY src/ src/
COPY --from=frontend /build/dist frontend/dist
COPY data/splits/ data/splits/
COPY data/tasks/ data/tasks/
COPY agents/ agents/
COPY runs/ runs/
COPY loop/ loop/

# The build's git SHA, passed in by the deploy workflow.
ARG CODE_SHA=unknown
ENV TAU2LOOP_CODE_SHA=$CODE_SHA

ENV DEMO_MODE=1 \
    BILLING=none \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "tau2_loop.serving.app:create_app", \
     "--factory", "--workers", "1", "--host", "0.0.0.0", "--port", "8080"]
