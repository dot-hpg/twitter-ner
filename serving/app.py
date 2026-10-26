"""REST API for Twitter NER: POST /ner, plus /health (liveness) and /ready (readiness) probes for Kubernetes."""
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from serving.ner import NerModel

MODEL_DIR = os.environ.get("MODEL_DIR", "models/bert_ner")
MAX_BATCH = int(os.environ.get("MAX_BATCH", "32"))
state = {}


@asynccontextmanager
async def lifespan(app):
    t0 = time.time()
    state["model"] = NerModel(MODEL_DIR)
    state["model"].predict(["warm up"])                    # build the TF graph before serving traffic
    state["load_seconds"] = round(time.time() - t0, 1)
    yield
    state.clear()


app = FastAPI(title="Twitter NER API", version="1.0.0", lifespan=lifespan,
              description="Fine-tuned bert-base-uncased tagging 10 entity types in tweets (WNUT-16).")


class NerRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, examples=[["Watching Stranger Things on Netflix in London"]])


class Entity(BaseModel):
    text: str
    type: str
    start: int
    end: int


class NerResult(BaseModel):
    words: list[str]
    tags: list[str]
    entities: list[Entity]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if "model" not in state:
        raise HTTPException(503, "model not loaded")
    return {"status": "ready", "model_dir": MODEL_DIR, "load_seconds": state["load_seconds"]}


@app.post("/ner", response_model=list[NerResult])
def ner(req: NerRequest):
    if len(req.texts) > MAX_BATCH:
        raise HTTPException(413, f"at most {MAX_BATCH} texts per request")
    if "model" not in state:
        raise HTTPException(503, "model not loaded")
    return state["model"].predict(req.texts)
