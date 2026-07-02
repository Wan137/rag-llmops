FROM python:3.11-slim

WORKDIR /app

# build-essential: some transitive wheels compile from source.
# libmagic1: unstructured's file-type detection needs it at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch's default PyPI wheel pulls the full CUDA stack (~2GB of unused nvidia-*
# packages) - this Space is CPU-only, so grab the CPU wheel instead.
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces runs the container as a non-root user with no fixed HOME, so give it
# a writable spot for the sqlite db, ChromaDB persistence, and the HF model cache.
RUN mkdir -p /app/data /app/mlruns && chmod -R 777 /app
ENV HOME=/app \
    HF_HOME=/app/.cache/huggingface \
    RAG_CHROMA_PERSIST_DIR=/app/data/chroma_db \
    MLFLOW_TRACKING_URI=/app/mlruns

EXPOSE 7860

# single worker - each gunicorn worker would load its own copy of the embedding
# model, and this is sized for portfolio-scale traffic, not concurrency.
CMD python manage.py migrate --noinput && \
    gunicorn backend.wsgi:application --bind 0.0.0.0:7860 --workers 1 --timeout 120
