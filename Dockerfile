# Twitter NER inference API (fine-tuned bert-base-uncased, TensorFlow CPU)
# Build from the project root after training:  docker build -t twitter-ner-api:1.0 .

FROM python:3.11-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY serving/requirements.txt /tmp/requirements.txt
RUN pip install --prefix=/install -r /tmp/requirements.txt

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 TF_USE_LEGACY_KERAS=1 \
    MODEL_DIR=/app/models/bert_ner HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY --from=deps /install /usr/local
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY serving/ serving/
COPY models/bert_ner/ models/bert_ner/
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready', timeout=4)"
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
