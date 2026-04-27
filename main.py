"""
Main execution script for PakSentinel.
Runs the end-to-end assignment pipeline with real artifacts and no placeholder inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.data_lake_manager import DataLakeManager
from src.data_sourcing import DataSourceManager
from src.ml_models import LogisticRegressionClassifier, MultinomialNaiveBayes, PolynomialLRClassifier
from src.mlflow_integration import ExperimentAblationStudy, MLFlowManager
from src.ngram_language_models import LanguageModelClassifier, NGramAnalyzer
from src.nlp_pipeline import NLPPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PakSentinelPipeline:
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = Path("./models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, object] = {}
        self.class_weights: Dict[str, float] | None = None

    def _save_json(self, name: str, payload: Dict):
        path = self.output_dir / name
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        return str(path)

    def run_task1_data_sourcing(self) -> pd.DataFrame:
        manager = DataSourceManager(data_dir="./data/raw")
        combined_df, stats = manager.combine_datasets(min_samples=5000)
        combined_df, imbalance_meta = manager.handle_class_imbalance(combined_df, method="class_weighted")
        reliability_report = manager.generate_reliability_report(combined_df, stats)
        reliability_report["imbalance_handling"] = imbalance_meta
        self.class_weights = imbalance_meta.get("class_weights")

        combined_df.to_parquet(self.output_dir / "task1_combined_dataset.parquet", index=False)
        self._save_json("task1_reliability_report.json", reliability_report)
        combined_df.to_parquet("./data/raw/combined_dataset.parquet", index=False)

        self.results["task1"] = {
            "dataset_size": int(len(combined_df)),
            "class_distribution": combined_df["label"].value_counts().to_dict(),
            "duplicate_rate": stats["duplicate_rate"],
            "class_weight_strategy": self.class_weights,
        }
        return combined_df

    def run_task2_storage(self, df: pd.DataFrame, processed_df: pd.DataFrame | None = None) -> DataLakeManager:
        manager = DataLakeManager(backend="local", base_path="./data/lake")
        raw_info = manager.upload_raw(df, source_name="combined_sources", description="Unified PakSentinel dataset")

        upload_summary = {"raw_upload": raw_info}
        if processed_df is not None:
            processed_info = manager.upload_processed(
                processed_df,
                name="processed_training_dataset",
                vectorizer_type="mixed",
                preprocessing_steps=["clean", "tokenize", "custom_stopwords"],
            )
            upload_summary["processed_upload"] = processed_info

        self.results["task2"] = upload_summary
        return manager

    def run_task3_nlp_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        pipeline = NLPPipeline(output_dir=str(self.output_dir))
        results, processed_df = pipeline.process_dataset(df, "text", "label")
        self._save_json("task3_pipeline_results.json", results)
        processed_df.to_parquet(self.output_dir / "task3_processed_data.parquet", index=False)
        self.results["task3"] = {
            "recommended_tokenizer": results["tokenizer_comparison"]["recommendation"],
            "stopword_f1_delta": results["stopword_analysis"]["f1_delta_custom_minus_default"],
            "word2vec_tsne_path": results["word2vec"]["tsne_path"],
        }
        return processed_df

    def run_task4_ngram_models(self, df: pd.DataFrame) -> Dict:
        train_df, test_df = train_test_split(
            df,
            test_size=min(100, max(60, int(len(df) * 0.1))),
            random_state=42,
            stratify=df["label"],
        )

        lm_classifier = LanguageModelClassifier(n=3, smoothing="kneser-ney")
        lm_stats = lm_classifier.train(train_df["text"].tolist(), train_df["label"].tolist())
        eval_results = lm_classifier.evaluate(test_df["text"].tolist(), test_df["label"].tolist())
        analysis = NGramAnalyzer.analyze_dataset(train_df["text"].tolist(), train_df["label"].tolist())
        comparison = NGramAnalyzer.compare_ngrams(analysis)

        task4_payload = {
            "language_model_stats": lm_stats,
            "heldout_metrics": eval_results,
            "top_ngrams": {key: [(list(ngram), count) for ngram, count in value] for key, value in comparison.items()},
        }
        self._save_json("task4_ngram_results.json", task4_payload)
        self.results["task4"] = {
            "accuracy": eval_results["accuracy"],
            "f1_score": eval_results["f1_score"],
        }
        return task4_payload

    def _plot_roc_curves(self, roc_data: Dict, save_path: Path):
        plt.figure(figsize=(8, 6))
        for label, metrics in roc_data.items():
            plt.plot(metrics["fpr"], metrics["tpr"], label=f"{label} AUC={metrics['auc']:.3f}")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Per-class ROC Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def _save_model_bundle(self, bundle: Dict):
        bundle_path = self.models_dir / "production_bundle.pkl"
        with open(bundle_path, "wb") as handle:
            pickle.dump(bundle, handle)
        return bundle_path

    def run_task5_ml_models(self, df: pd.DataFrame) -> Dict:
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df["cleaned_text"].fillna(df["text"]).tolist(),
            df["label"].tolist(),
            test_size=0.2,
            random_state=42,
            stratify=df["label"].tolist(),
        )

        bow_vectorizer = CountVectorizer(max_features=5000)
        X_train_bow = bow_vectorizer.fit_transform(X_train_text)
        X_test_bow = bow_vectorizer.transform(X_test_text)

        tfidf_vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True, smooth_idf=True)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
        X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

        class_weights = self.class_weights if self.class_weights else "balanced"
        results: Dict[str, object] = {}

        nb = MultinomialNaiveBayes(alpha=1.0)
        nb.fit(X_train_bow, np.array(y_train), feature_names=bow_vectorizer.get_feature_names_out().tolist())
        nb_predictions = nb.predict(X_test_bow)
        nb_probabilities = nb.predict_proba(X_test_bow)
        nb_alpha = nb.alpha_sensitivity_analysis(X_test_bow, np.array(y_test), [0.01, 0.1, 0.5, 1.0, 2.0, 5.0])

        misclassified = []
        for text, truth, pred in zip(X_test_text, y_test, nb_predictions):
            if truth != pred:
                misclassified.append(
                    {
                        "text": text[:300],
                        "true_label": truth,
                        "predicted_label": pred,
                        "error_type": f"{truth}_to_{pred}",
                    }
                )
            if len(misclassified) == 30:
                break

        results["naive_bayes"] = {
            "accuracy": float(accuracy_score(y_test, nb_predictions)),
            "precision": float(precision_score(y_test, nb_predictions, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, nb_predictions, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, nb_predictions, average="weighted", zero_division=0)),
            "classification_report": classification_report(y_test, nb_predictions, output_dict=True, zero_division=0),
            "alpha_sensitivity": nb_alpha,
            "misclassified_samples": misclassified,
        }

        roc_bundle = {}
        logistic_artifacts = {}
        for regularization in ("l1", "l2", "elasticnet"):
            start = time.perf_counter()
            lr = LogisticRegressionClassifier(regularization=regularization, C=1.0, class_weight=class_weights)
            lr.fit(X_train_tfidf, np.array(y_train), feature_names=tfidf_vectorizer.get_feature_names_out().tolist())
            elapsed = time.perf_counter() - start
            eval_metrics = lr.evaluate(X_test_tfidf, np.array(y_test))
            eval_metrics["training_time_sec"] = float(elapsed)
            eval_metrics["top_features"] = {
                label: lr.get_top_features(label, top_n=20) for label in lr.classes
            }
            results[f"logistic_regression_{regularization}"] = eval_metrics
            roc_bundle[regularization] = lr.compute_roc_data(X_test_tfidf, np.array(y_test))
            logistic_artifacts[regularization] = lr

        roc_plot_path = self.output_dir / "task5_lr_roc_curves.png"
        # Assignment asks one figure with all three variants, so plot per class/variant labels.
        plt.figure(figsize=(10, 7))
        for regularization, roc_data in roc_bundle.items():
            for label, metrics in roc_data.items():
                plt.plot(metrics["fpr"], metrics["tpr"], label=f"{regularization}:{label} AUC={metrics['auc']:.2f}")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Logistic Regression ROC Curves")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(roc_plot_path, dpi=150)
        plt.close()

        poly = PolynomialLRClassifier(n_components=2)
        poly.fit(X_train_tfidf, np.array(y_train))
        poly_results = poly.evaluate_degrees(X_test_tfidf, np.array(y_test))
        results["polynomial_lr"] = poly_results
        for degree in (1, 2, 3):
            poly.plot_decision_boundaries(
                degree=degree,
                y=np.array(y_train),
                save_path=str(self.output_dir / f"task5_poly_degree_{degree}.png"),
            )
        results["polynomial_lr"]["degree_2_full_feature_space_size"] = int(
            (tfidf_vectorizer.transform(X_train_text).shape[1] + 2) * (tfidf_vectorizer.transform(X_train_text).shape[1] + 1) / 2
        )
        results["polynomial_lr"]["alternative_non_linear_approach"] = "Kernel SVM on reduced TF-IDF features"

        best_lr = max(
            ("l1", "l2", "elasticnet"),
            key=lambda reg: results[f"logistic_regression_{reg}"]["f1_weighted"],
        )
        production_bundle = {
            "model_name": f"logistic_regression_{best_lr}",
            "model": logistic_artifacts[best_lr],
            "vectorizer": tfidf_vectorizer,
            "labels": sorted(df["label"].unique().tolist()),
            "retrieval_corpus": df[["text", "cleaned_text", "label", "source_id"]].fillna("").to_dict(orient="records"),
            "retrieval_matrix": tfidf_vectorizer.transform(df["cleaned_text"].fillna(df["text"]).tolist()),
            "metadata": {
                "stage": "production",
                "version": "1.0.0",
                "f1_score": results[f"logistic_regression_{best_lr}"]["f1_weighted"],
                "class_weights": class_weights,
                "created_at": pd.Timestamp.now().isoformat(),
                "source_count": int(df["source_id"].nunique()),
                "roc_plot": str(roc_plot_path),
            },
        }
        bundle_path = self._save_model_bundle(production_bundle)
        with open(self.models_dir / "production_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(production_bundle["metadata"] | {"model_name": production_bundle["model_name"], "bundle_path": str(bundle_path)}, handle, indent=2)

        self._save_json("task5_ml_results.json", results)
        self.results["task5"] = {
            "naive_bayes_f1": results["naive_bayes"]["f1_score"],
            "best_logistic_regression": best_lr,
            "best_lr_f1": results[f"logistic_regression_{best_lr}"]["f1_weighted"],
            "production_bundle": str(bundle_path),
        }
        return results

    def run_task6_mlflow(self, df: pd.DataFrame) -> Dict:
        tracking_uri = "sqlite:///mlflow_data/mlflow.db"
        manager = MLFlowManager(tracking_uri=tracking_uri, experiment_name="PakSentinel")
        ablation = ExperimentAblationStudy(manager)
        ablation_results = ablation.run_ablation_study(df, text_col="cleaned_text" if "cleaned_text" in df.columns else "text", label_col="label")
        parallel_plot = ablation.get_parallel_coordinates_plot(self.output_dir / "task6_parallel_coordinates.png")

        registry_summary = manager.register_best_family_models(ablation_results)
        self.results["task6"] = {
            "tracking_uri": tracking_uri,
            "ablation_runs": len(ablation_results),
            "parallel_coordinates_plot": str(parallel_plot),
            "registry_summary": registry_summary,
        }
        return self.results["task6"]

    def run_task7_api_tests(self):
        self.results["task7"] = {
            "api_ready": (self.models_dir / "production_bundle.pkl").exists(),
            "bundle_path": str(self.models_dir / "production_bundle.pkl"),
            "expected_app_module": "api.app:app",
        }

    def generate_summary_report(self) -> Dict:
        report = {
            "pipeline_name": "PakSentinel",
            "execution_timestamp": pd.Timestamp.now().isoformat(),
            "task_results": self.results,
        }
        self._save_json("summary_report.json", report)
        return report

    def run_all_tasks(self, skip_tasks: List[str] | None = None) -> Dict:
        skip_tasks = skip_tasks or []
        df = None
        processed_df = None

        if "task1" not in skip_tasks:
            df = self.run_task1_data_sourcing()
        if df is None:
            task1_path = self.output_dir / "task1_combined_dataset.parquet"
            df = pd.read_parquet(task1_path)

        if "task3" not in skip_tasks:
            processed_df = self.run_task3_nlp_pipeline(df)
        if processed_df is None and (self.output_dir / "task3_processed_data.parquet").exists():
            processed_df = pd.read_parquet(self.output_dir / "task3_processed_data.parquet")
        working_df = processed_df if processed_df is not None else df

        if "task2" not in skip_tasks:
            self.run_task2_storage(df, processed_df=working_df)
        if "task4" not in skip_tasks:
            self.run_task4_ngram_models(working_df)
        if "task5" not in skip_tasks:
            self.run_task5_ml_models(working_df)
        if "task6" not in skip_tasks:
            self.run_task6_mlflow(working_df)
        if "task7" not in skip_tasks:
            self.run_task7_api_tests()

        return self.generate_summary_report()


def main():
    parser = argparse.ArgumentParser(description="PakSentinel pipeline")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--skip-task", action="append")
    args = parser.parse_args()

    pipeline = PakSentinelPipeline(output_dir=args.output_dir)
    report = pipeline.run_all_tasks(skip_tasks=args.skip_task or [])
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
