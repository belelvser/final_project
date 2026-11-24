FROM python:3.13-slim-bookworm@sha256:f0fa218d5b06f9b9235024fa061cde2cb141f726de52134471053d2d9b5324a3
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY main.py config.json pytest.ini ./
COPY churn_pipeline/ ./churn_pipeline/
COPY data/full.parquet data/sample.parquet data/provenance.json ./data/
COPY tests/ ./tests/
ENTRYPOINT ["python", "main.py"]
CMD ["--mode", "demo", "--output-dir", "/outputs"]
