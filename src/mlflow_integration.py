"""
Task 6: MLflow experiment tracking with real preprocessing ablations.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .nlp_pipeline import NLPPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLFlowManager:
    """Manage experiment logging and model promotion."""

    def __init__(self, tracking_uri: str = "sqlite:///mlflow_data/mlflow.db", experiment_name: str = "PakSentinel"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        self.experiment_id = experiment.experiment_id if experiment else None

    def log_model_training_run(
        self,
        run_name: str,
        model: Any,
        X_train,
        X_test,
        y_train: List[str],
        y_test: List[str],
        y_pred: List[str],
        y_proba: np.ndarray,
        params: Dict[str, Any],
        dataset_sources: List[str],
        tokenizer: str,
        stopword_list: str,
        normalization: str,
        vectorizer_settings: Dict[str, Any],
        training_time: float,
        vocabulary: Optional[List[str]] = None,
        artifact_dir: Optional[Path] = None,
    ) -> str:
        artifact_dir = artifact_dir or Path(tempfile.mkdtemp(prefix="mlflow_run_"))
        classes = sorted(set(y_test))
        precision, recall, f1_values, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=classes, zero_division=0
        )
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(
                {
                    "dataset_sources": ",".join(dataset_sources),
                    "train_size": len(y_train),
                    "test_size": len(y_test),
                    "tokenizer": tokenizer,
                    "stopword_list": stopword_list,
                    "normalization_method": normalization,
                    "vectorizer_settings": json.dumps(vectorizer_settings, sort_keys=True),
                    "model_type": params.get("model_type", "unknown"),
                }
            )
            mlflow.log_params({key: value for key, value in params.items() if key != "model_type"})

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "weighted_f1": weighted_f1,
                "training_time_sec": float(training_time),
            }
            for label, p_value, r_value, f_value in zip(classes, precision, recall, f1_values):
                metrics[f"{label}_precision"] = float(p_value)
                metrics[f"{label}_recall"] = float(r_value)
                metrics[f"{label}_f1"] = float(f_value)

            if y_proba is not None and len(classes) > 1:
                class_to_idx = {label: idx for idx, label in enumerate(classes)}
                y_true_matrix = np.zeros((len(y_test), len(classes)))
                for row_idx, label in enumerate(y_test):
                    y_true_matrix[row_idx, class_to_idx[label]] = 1
                metrics["roc_auc_macro"] = float(
                    roc_auc_score(y_true_matrix, y_proba, multi_class="ovr", average="macro")
                )

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            artifact_dir.mkdir(parents=True, exist_ok=True)
            cm = confusion_matrix(y_test, y_pred, labels=classes)
            self._save_confusion_matrix(cm, classes, artifact_dir / "confusion_matrix.png")
            self._save_classification_report(y_test, y_pred, artifact_dir / "classification_report.json")
            if vocabulary is not None:
                with open(artifact_dir / "tfidf_vocabulary.json", "w", encoding="utf-8") as handle:
                    json.dump(vocabulary, handle, indent=2)

            mlflow.log_artifacts(str(artifact_dir))
            shutil.rmtree(artifact_dir, ignore_errors=True)
            return run.info.run_id

    def _save_confusion_matrix(self, cm: np.ndarray, classes: List[str], save_path: Path):
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_positions = np.arange(len(classes))
        plt.xticks(tick_positions, classes, rotation=45)
        plt.yticks(tick_positions, classes)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def _save_classification_report(self, y_true: List[str], y_pred: List[str], save_path: Path):
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        with open(save_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    def register_model(self, run_id: str, model_name: str, stage: str = "Staging") -> Dict:
        client = mlflow.tracking.MlflowClient()
        registered = mlflow.register_model(f"runs:/{run_id}/model", model_name)
        client.transition_model_version_stage(name=model_name, version=registered.version, stage=stage)
        return {"model_name": model_name, "version": registered.version, "stage": stage}

    def promote_model(self, model_name: str, candidate_run_id: str, min_relative_gain: float = 0.01) -> bool:
        client = mlflow.tracking.MlflowClient()
        candidate_run = mlflow.get_run(candidate_run_id)
        candidate_f1 = candidate_run.data.metrics.get("weighted_f1", 0.0)

        production_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not production_versions:
            latest = client.get_latest_versions(model_name, stages=["Staging"])
            if latest:
                client.transition_model_version_stage(model_name, latest[0].version, "Production")
                return True
            return False

        current_prod = production_versions[0]
        current_f1 = mlflow.get_run(current_prod.run_id).data.metrics.get("weighted_f1", 0.0)
        improvement = candidate_f1 - current_f1
        if improvement >= max(0.01, current_f1 * min_relative_gain):
            client.transition_model_version_stage(model_name, current_prod.version, "Archived")
            latest = client.get_latest_versions(model_name, stages=["Staging"])
            if latest:
                client.transition_model_version_stage(model_name, latest[0].version, "Production")
                return True
        return False


class ExperimentAblationStudy:
    """Run the required preprocessing ablation study."""

    def __init__(self, mlflow_manager: MLFlowManager):
        self.mlflow_manager = mlflow_manager
        self.completed_runs: List[Dict[str, Any]] = []

    def _apply_config(self, texts: List[str], pipeline: NLPPipeline, config: Dict[str, Any]) -> List[str]:
        processed_docs = []
        for text in texts:
            cleaned = pipeline.cleaner.clean(text)
            tokens = pipeline.tokenizer_comparison.tokenize_nltk(cleaned)
            if config["stopwords"] == "custom":
                tokens = pipeline.stopword_manager.remove_stopwords(tokens, use_custom=True)[0]
            elif config["stopwords"] == "default":
                tokens = pipeline.stopword_manager.remove_stopwords(tokens, use_custom=False)[0]

            if config["normalization"] == "porter":
                tokens = pipeline.normalization.stem_porter(tokens)
            elif config["normalization"] == "snowball":
                tokens = pipeline.normalization.stem_snowball(tokens)
            elif config["normalization"] == "wordnet":
                tokens = pipeline.normalization.lemmatize_wordnet(tokens)

            tokens = [token for token in tokens if len(token) >= config["min_token_length"]]
            processed_docs.append(" ".join(tokens))
        return processed_docs

    def run_ablation_study(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "label",
        configurations: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if configurations is None:
            configurations = [
                {"name": "baseline", "stopwords": "default", "normalization": "none", "min_token_length": 1, "max_features": 5000},
                {"name": "custom_stopwords", "stopwords": "custom", "normalization": "none", "min_token_length": 1, "max_features": 5000},
                {"name": "porter_stemming", "stopwords": "custom", "normalization": "porter", "min_token_length": 1, "max_features": 5000},
                {"name": "wordnet_lemma", "stopwords": "custom", "normalization": "wordnet", "min_token_length": 1, "max_features": 5000},
                {"name": "min_len_3", "stopwords": "custom", "normalization": "wordnet", "min_token_length": 3, "max_features": 5000},
                {"name": "reduced_vocab", "stopwords": "custom", "normalization": "wordnet", "min_token_length": 3, "max_features": 2000},
            ]

        pipeline = NLPPipeline(output_dir="./output")
        texts = df[text_col].fillna("").astype(str).tolist()
        labels = df[label_col].tolist()
        dataset_sources = sorted(df["source_id"].astype(str).unique().tolist()) if "source_id" in df.columns else ["unknown"]
        self.completed_runs = []

        for config in configurations:
            processed_docs = self._apply_config(texts, pipeline, config)
            X_train, X_test, y_train, y_test = train_test_split(
                processed_docs, labels, test_size=0.2, random_state=42, stratify=labels
            )

            vectorizer = TfidfVectorizer(max_features=config["max_features"], sublinear_tf=True, smooth_idf=True)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            start = time.perf_counter()
            model = LogisticRegression(max_iter=1000, class_weight="balanced")
            model.fit(X_train_vec, y_train)
            elapsed = time.perf_counter() - start

            predictions = model.predict(X_test_vec)
            probabilities = model.predict_proba(X_test_vec)

            run_id = self.mlflow_manager.log_model_training_run(
                run_name=config["name"],
                model=model,
                X_train=X_train_vec,
                X_test=X_test_vec,
                y_train=y_train,
                y_test=y_test,
                y_pred=predictions,
                y_proba=probabilities,
                params={"model_type": "logistic_regression", **config},
                dataset_sources=dataset_sources,
                tokenizer="nltk",
                stopword_list=config["stopwords"],
                normalization=config["normalization"],
                vectorizer_settings={"max_features": config["max_features"], "sublinear_tf": True, "smooth_idf": True},
                training_time=elapsed,
                vocabulary=vectorizer.get_feature_names_out().tolist(),
            )

            self.completed_runs.append(
                {
                    "run_id": run_id,
                    "run_name": config["name"],
                    "family": "preprocessing_ablation",
                    "model_family": "logistic_regression",
                    "config": config,
                    "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
                }
            )

        return self.completed_runs

    def get_parallel_coordinates_plot(self, save_path: Path) -> Path:
        if not self.completed_runs:
            raise ValueError("No completed ablation runs to plot.")

        rows = []
        for item in self.completed_runs:
            config = item["config"]
            rows.append(
                {
                    "run_name": item["run_name"],
                    "stopwords": 1 if config["stopwords"] == "custom" else 0,
                    "normalization": {"none": 0, "porter": 1, "snowball": 2, "wordnet": 3}[config["normalization"]],
                    "min_token_length": config["min_token_length"],
                    "max_features": config["max_features"],
                    "f1_weighted": item["weighted_f1"],
                }
            )

        df = pd.DataFrame(rows)
        plt.figure(figsize=(11, 6))
        parallel_coordinates(df, class_column="run_name", cols=["stopwords", "normalization", "min_token_length", "max_features", "f1_weighted"])
        plt.ylabel("Scaled / raw value")
        plt.title("Preprocessing Ablation Parallel Coordinates")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        return save_path

    def register_best_family_models(self, completed_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not completed_runs:
            return {}

        best_run = max(completed_runs, key=lambda item: item["weighted_f1"])
        registration = self.mlflow_manager.register_model(best_run["run_id"], "PakSentinel-LogisticRegression", stage="Staging")
        promoted = self.mlflow_manager.promote_model("PakSentinel-LogisticRegression", best_run["run_id"])
        return {"best_run": best_run, "registration": registration, "promoted_to_production": promoted}
