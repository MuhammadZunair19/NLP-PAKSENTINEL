"""
Task 7: FastAPI inference system backed by saved training artifacts.
"""

import json
import logging
import os
import pickle
import time
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.nlp_pipeline import NLPPipeline


LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("paksentinel_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler(LOG_DIR / "api.log", maxBytes=2_000_000, backupCount=3)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

limiter = Limiter(key_func=get_remote_address)


class PreprocessRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000)
    steps: List[str] = Field(default=["clean", "tokenize", "stopwords"])


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000)
    model_type: str = Field(default="production")


class BatchClassifyRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)
    model_type: str = Field(default="production")

    @field_validator("texts")
    @classmethod
    def validate_text_lengths(cls, texts: List[str]) -> List[str]:
        for text in texts:
            if not 10 <= len(text) <= 10000:
                raise ValueError("Each text must be between 10 and 10,000 characters.")
        return texts


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=10000)
    top_k: int = Field(default=5, ge=1, le=20)


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    model_stage: str
    f1_score: float
    load_timestamp: str


class ModelManager:
    def __init__(self, bundle_path: str = "./models/production_bundle.pkl", metadata_path: str = "./models/production_metadata.json"):
        self.bundle_path = Path(bundle_path)
        self.metadata_path = Path(metadata_path)
        self.pipeline = NLPPipeline(output_dir="./output")
        self.bundle = None
        self.metadata: Dict[str, object] = {}
        self.load_timestamp = None

    def load_models(self):
        if not self.bundle_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {self.bundle_path}. Run the training pipeline before starting the API."
            )
        with open(self.bundle_path, "rb") as handle:
            self.bundle = pickle.load(handle)
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as handle:
                self.metadata = json.load(handle)
        else:
            self.metadata = self.bundle.get("metadata", {})
        self.load_timestamp = self.metadata.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S")

    @property
    def model_name(self) -> str:
        return str(self.bundle["model_name"])

    def preprocess_text(self, text: str, steps: List[str]) -> Tuple[List[str], List[str], float]:
        start = time.perf_counter()
        if "clean" in steps:
            text = self.pipeline.cleaner.clean(text)
        tokens = self.pipeline.tokenizer_comparison.tokenize_nltk(text) if "tokenize" in steps else text.split()
        removed: List[str] = []
        if "stopwords" in steps:
            tokens, removed = self.pipeline.stopword_manager.remove_stopwords(tokens, use_custom=True)
        elapsed = (time.perf_counter() - start) * 1000
        return tokens, removed, elapsed

    def classify(self, text: str) -> Dict:
        start = time.perf_counter()
        cleaned = self.pipeline.cleaner.clean(text)
        vectorizer = self.bundle["vectorizer"]
        model = self.bundle["model"]
        features = vectorizer.transform([cleaned])
        probabilities = model.predict_proba(features)[0]
        prediction = model.predict(features)[0]
        class_probabilities = {
            str(label): float(prob) for label, prob in zip(model.classes, probabilities)
        }

        predicted_binary_model = model.models[prediction]
        scaled_features = model.scaler.transform(features.toarray())
        contribution_scores = scaled_features[0] * predicted_binary_model.coef_[0]
        feature_names = np.array(model.feature_names)
        nonzero_indices = np.where(features.toarray()[0] > 0)[0]
        top_indices = sorted(nonzero_indices, key=lambda idx: abs(contribution_scores[idx]), reverse=True)[:5]
        top_features = [(str(feature_names[idx]), float(contribution_scores[idx])) for idx in top_indices]

        elapsed = (time.perf_counter() - start) * 1000
        return {
            "prediction": prediction,
            "confidence": float(max(class_probabilities.values())),
            "class_probabilities": class_probabilities,
            "top_contributing_features": top_features,
            "processing_time_ms": elapsed,
        }

    def retrieve_similar(self, query: str, top_k: int) -> List[Dict]:
        cleaned = self.pipeline.cleaner.clean(query)
        vectorizer = self.bundle["vectorizer"]
        retrieval_matrix = self.bundle["retrieval_matrix"]
        corpus = self.bundle["retrieval_corpus"]
        query_vec = vectorizer.transform([cleaned])
        scores = (query_vec @ retrieval_matrix.T).toarray()[0]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            record = corpus[idx]
            results.append(
                {
                    "claim": record.get("text") or record.get("cleaned_text"),
                    "similarity_score": float(scores[idx]),
                    "source": record.get("source_id", "unknown"),
                    "label": record.get("label", "unknown"),
                }
            )
        return results

    def performance_payload(self) -> Dict:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow_data/mlflow.db")
        current_metrics = [
            {
                "metric_name": "weighted_f1",
                "value": float(self.metadata.get("f1_score", 0.0)),
                "timestamp": str(self.load_timestamp),
            }
        ]
        version_history = [
            {
                "version": str(self.metadata.get("version", "unknown")),
                "stage": str(self.metadata.get("stage", "production")),
                "f1_score": float(self.metadata.get("f1_score", 0.0)),
                "timestamp": str(self.load_timestamp),
            }
        ]

        try:
            mlflow.set_tracking_uri(tracking_uri)
            experiment = mlflow.get_experiment_by_name("PakSentinel")
            if experiment is not None:
                client = mlflow.tracking.MlflowClient()
                runs = client.search_runs([experiment.experiment_id], order_by=["metrics.weighted_f1 DESC"], max_results=5)
                version_history = [
                    {
                        "run_id": run.info.run_id,
                        "version": run.info.run_name,
                        "stage": "tracked",
                        "f1_score": float(run.data.metrics.get("weighted_f1", 0.0)),
                        "timestamp": run.info.start_time,
                    }
                    for run in runs
                ]
                if runs:
                    current_metrics = [
                        {
                            "metric_name": key,
                            "value": float(value),
                            "timestamp": runs[0].info.start_time,
                        }
                        for key, value in runs[0].data.metrics.items()
                    ]
        except Exception as exc:
            logger.warning("MLflow metrics unavailable: %s", exc)

        return {"current_metrics": current_metrics, "version_history": version_history}


model_manager: ModelManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager
    model_manager = ModelManager()
    model_manager.load_models()
    yield
    model_manager = None


app = FastAPI(
    title="PakSentinel API",
    description="Misinformation detection and retrieval service",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s status=%s time_ms=%.2f", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.get("/health", response_model=HealthResponse)
@limiter.limit("100/minute")
async def health(request: Request):
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized.")
    return HealthResponse(
        status="healthy",
        model_name=model_manager.model_name,
        model_version=str(model_manager.metadata.get("version", "unknown")),
        model_stage=str(model_manager.metadata.get("stage", "production")),
        f1_score=float(model_manager.metadata.get("f1_score", 0.0)),
        load_timestamp=str(model_manager.load_timestamp),
    )


@app.post("/preprocess")
@limiter.limit("100/minute")
async def preprocess(request: Request, payload: PreprocessRequest):
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized.")
    tokens, removed, elapsed = model_manager.preprocess_text(payload.text, payload.steps)
    return {
        "original_text": payload.text,
        "tokens": tokens,
        "removed_stopwords": removed,
        "processing_time_ms": elapsed,
        "steps_applied": payload.steps,
    }


@app.post("/classify")
@limiter.limit("100/minute")
async def classify(request: Request, payload: ClassifyRequest):
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized.")
    result = model_manager.classify(payload.text)
    return {"text": payload.text, **result}


@app.post("/classify/batch")
@limiter.limit("10/minute")
async def classify_batch(request: Request, payload: BatchClassifyRequest):
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized.")
    start = time.perf_counter()
    predictions = [model_manager.classify(text) for text in payload.texts]
    elapsed = (time.perf_counter() - start) * 1000
    return {
        "predictions": [item["prediction"] for item in predictions],
        "class_probabilities": [item["class_probabilities"] for item in predictions],
        "total_time_ms": elapsed,
        "avg_time_per_sample_ms": elapsed / len(payload.texts),
    }


@app.post("/retrieve/similar")
@limiter.limit("50/minute")
async def retrieve_similar(request: Request, payload: RetrievalRequest):
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized.")
    return {
        "query": payload.query,
        "top_k": payload.top_k,
        "results": model_manager.retrieve_similar(payload.query, payload.top_k),
    }


@app.get("/model/performance")
@limiter.limit("50/minute")
async def model_performance(request: Request):
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized.")
    return model_manager.performance_payload()


@app.get("/")
async def root():
    return {"name": "PakSentinel", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
