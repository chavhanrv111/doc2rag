FROM python:3.12-slim

ARG EXTRAS=docling
ENV PYTHONUNBUFFERED=1 \
    DOC2RAG_OUTPUT_DIR=/data/out \
    HF_HOME=/models

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[${EXTRAS}]"

VOLUME ["/data", "/models"]
ENTRYPOINT ["doc2rag"]
CMD ["--help"]
