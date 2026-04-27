import asyncio
import json
import pickle
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sklearn.feature_extraction.text import TfidfVectorizer

from api.app import app
from src.ml_models import LogisticRegressionClassifier
from src.nlp_pipeline import NLPPipeline


@pytest.fixture(scope="session", autouse=True)
def build_test_bundle():
    models_dir = Path("./models")
    models_dir.mkdir(parents=True, exist_ok=True)

    texts = [
        "official government policy update from ministry of health",
        "authentic court verdict announced by supreme court",
        "verified election commission statement on polling",
        "viral fake cure claim spreading on social media",
        "false rumor about celebrity death shared widely",
        "fabricated scandal accusation without evidence",
        "satire piece mocking political promises with humor",
        "parody headline about impossible policy success",
        "comic exaggeration of budget announcement",
    ] * 20
    labels = (["Real"] * 3 + ["Fake"] * 3 + ["Satire"] * 3) * 20

    pipeline = NLPPipeline(output_dir="./output")
    cleaned = [pipeline.cleaner.clean(text) for text in texts]
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(cleaned)
    model = LogisticRegressionClassifier(regularization="l2", class_weight="balanced")
    model.fit(X, labels, feature_names=vectorizer.get_feature_names_out().tolist())

    bundle = {
        "model_name": "logistic_regression_l2",
        "model": model,
        "vectorizer": vectorizer,
        "retrieval_corpus": [
            {"text": text, "cleaned_text": cleaned_text, "label": label, "source_id": "test_source"}
            for text, cleaned_text, label in zip(texts, cleaned, labels)
        ],
        "retrieval_matrix": vectorizer.transform(cleaned),
        "metadata": {
            "version": "test",
            "stage": "production",
            "f1_score": 0.95,
            "created_at": "2026-04-27T00:00:00",
        },
    }
    with open(models_dir / "production_bundle.pkl", "wb") as handle:
        pickle.dump(bundle, handle)
    with open(models_dir / "production_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(bundle["metadata"] | {"model_name": bundle["model_name"]}, handle)


@pytest.fixture
def client(build_test_bundle):
    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoint:
    def test_health_check_success(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_name"] == "logistic_regression_l2"
        assert data["f1_score"] == 0.95


class TestPreprocessEndpoint:
    def test_preprocess_success(self, client: TestClient):
        response = client.post(
            "/preprocess",
            json={"text": "This is a test sample text for preprocessing", "steps": ["clean", "tokenize", "stopwords"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["tokens"], list)
        assert isinstance(data["removed_stopwords"], list)
        assert data["processing_time_ms"] < 1000

    def test_preprocess_length_validation(self, client: TestClient):
        response = client.post("/preprocess", json={"text": "short", "steps": ["clean"]})
        assert response.status_code == 422


class TestClassifyEndpoint:
    def test_classify_success(self, client: TestClient):
        response = client.post(
            "/classify",
            json={"text": "official ministry statement confirms verified policy update", "model_type": "production"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in {"Real", "Fake", "Satire"}
        assert abs(sum(data["class_probabilities"].values()) - 1.0) < 1e-6
        assert len(data["top_contributing_features"]) <= 5

    def test_classify_response_time(self, client: TestClient):
        start = time.perf_counter()
        response = client.post(
            "/classify",
            json={"text": "viral fake cure claim shared without evidence online", "model_type": "production"},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert response.status_code == 200
        assert elapsed_ms < 100


class TestBatchClassifyEndpoint:
    def test_batch_classify_success(self, client: TestClient):
        payload = {
            "texts": [
                "official election commission statement on polling process",
                "viral fake cure claim shared without evidence online",
                "satire article mocking campaign promises for laughs",
            ]
        }
        response = client.post("/classify/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 3
        assert data["avg_time_per_sample_ms"] <= data["total_time_ms"]

    def test_batch_classify_text_validation(self, client: TestClient):
        response = client.post("/classify/batch", json={"texts": ["short"]})
        assert response.status_code == 422


class TestRetrievalEndpoint:
    def test_retrieve_similar_success(self, client: TestClient):
        response = client.post("/retrieve/similar", json={"query": "official ministry health statement", "top_k": 3})
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 3
        assert all("similarity_score" in item for item in data["results"])


class TestPerformanceEndpoint:
    def test_performance_metrics_success(self, client: TestClient):
        response = client.get("/model/performance")
        assert response.status_code == 200
        data = response.json()
        assert "current_metrics" in data
        assert "version_history" in data


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, build_test_bundle):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            tasks = [
                client.post(
                    "/classify",
                    json={"text": f"official statement number {idx} about verified policy update", "model_type": "production"},
                )
                for idx in range(5)
            ]
            responses = await asyncio.gather(*tasks)
            assert all(response.status_code == 200 for response in responses)
