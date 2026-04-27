"""
Task 1: Data Sourcing & Reliability Assessment
Load reproducible source-backed datasets from local files and build a unified
Real/Fake/Satire corpus without synthetic or dummy fallbacks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CANONICAL_CLASSES = ("Real", "Fake", "Satire")


@dataclass(frozen=True)
class SourceConfig:
    name: str
    source_id: str
    paths: Tuple[Path, ...]
    text_columns: Tuple[str, ...]
    label_column: str
    label_mapping: Dict[str, str]
    date_columns: Tuple[str, ...] = ()


class DataReliabilityScorecard:
    """Evaluate each source against the assignment scorecard."""

    def __init__(self, source_name: str):
        self.source_name = source_name
        self.metrics = {
            "label_credibility": 0,
            "recency": 0,
            "pakistan_relevance": 0,
            "class_balance": 0,
            "language_consistency": 0,
        }
        self.details: Dict[str, object] = {}

    def evaluate(self, df: pd.DataFrame, **kwargs) -> Dict:
        source_lower = self.source_name.lower()

        if "liar" in source_lower:
            self.metrics["label_credibility"] = 5
        elif "isot" in source_lower:
            self.metrics["label_credibility"] = 4
        elif "covid" in source_lower:
            self.metrics["label_credibility"] = 4
        else:
            self.metrics["label_credibility"] = kwargs.get("label_credibility", 3)

        year_series = pd.to_numeric(df.get("year"), errors="coerce")
        if year_series.notna().any():
            avg_year = float(year_series.dropna().mean())
            if avg_year >= 2023:
                self.metrics["recency"] = 5
            elif avg_year >= 2021:
                self.metrics["recency"] = 4
            elif avg_year >= 2019:
                self.metrics["recency"] = 3
            else:
                self.metrics["recency"] = 2
        else:
            self.metrics["recency"] = kwargs.get("recency", 2)

        if any(term in source_lower for term in ("pakistan", "dawn", "geo", "ary")):
            self.metrics["pakistan_relevance"] = 5
        elif "covid" in source_lower:
            self.metrics["pakistan_relevance"] = 3
        else:
            self.metrics["pakistan_relevance"] = 2

        class_dist = df["label"].value_counts()
        ratio = class_dist.max() / class_dist.min() if len(class_dist) > 1 else float("inf")
        if ratio <= 1.25:
            self.metrics["class_balance"] = 5
        elif ratio <= 1.75:
            self.metrics["class_balance"] = 4
        elif ratio <= 2.5:
            self.metrics["class_balance"] = 3
        elif ratio <= 4.0:
            self.metrics["class_balance"] = 2
        else:
            self.metrics["class_balance"] = 1

        lengths = df["text"].fillna("").str.len()
        if lengths.mean() == 0:
            self.metrics["language_consistency"] = 1
        else:
            cv = float(lengths.std(ddof=0) / lengths.mean()) if lengths.mean() else 0.0
            if cv < 0.45:
                self.metrics["language_consistency"] = 5
            elif cv < 0.8:
                self.metrics["language_consistency"] = 4
            elif cv < 1.2:
                self.metrics["language_consistency"] = 3
            elif cv < 1.6:
                self.metrics["language_consistency"] = 2
            else:
                self.metrics["language_consistency"] = 1

        self.details = kwargs
        return self.to_dict()

    def to_dict(self) -> Dict:
        return {
            "source": self.source_name,
            "metrics": self.metrics,
            "average_score": float(np.mean(list(self.metrics.values()))),
            "details": self.details,
        }


class DataSourceManager:
    """Load, normalize, and combine assignment-approved local datasets."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.source_configs = self._build_source_configs()
        self.class_weights_: Optional[Dict[str, float]] = None

    def _build_source_configs(self) -> Dict[str, SourceConfig]:
        raw_dir = self.data_dir
        return {
            "liar": SourceConfig(
                name="LIAR",
                source_id="LIAR",
                paths=(
                    raw_dir / "liar" / "train.tsv",
                    raw_dir / "liar" / "valid.tsv",
                    raw_dir / "liar" / "test.tsv",
                ),
                text_columns=("statement",),
                label_column="label",
                label_mapping={
                    "true": "Real",
                    "mostly-true": "Real",
                    "half-true": "Real",
                    "barely-true": "Fake",
                    "false": "Fake",
                    "pants-fire": "Fake",
                },
            ),
            "isot": SourceConfig(
                name="ISOT",
                source_id="ISOT",
                paths=(
                    raw_dir / "isot" / "True.csv",
                    raw_dir / "isot" / "Fake.csv",
                    raw_dir / "isot" / "true.csv",
                    raw_dir / "isot" / "fake.csv",
                ),
                text_columns=("title", "text"),
                label_column="label",
                label_mapping={"true": "Real", "real": "Real", "fake": "Fake"},
                date_columns=("date",),
            ),
            "covid": SourceConfig(
                name="COVID-19",
                source_id="COVID-19",
                paths=(
                    raw_dir / "covid" / "covid_19_data.csv",
                    raw_dir / "covid" / "Constraint_Train.csv",
                    raw_dir / "covid" / "Constraint_Val.csv",
                ),
                text_columns=("text", "statement", "headlines"),
                label_column="label",
                label_mapping={
                    "0": "Real",
                    "1": "Fake",
                    "real": "Real",
                    "fake": "Fake",
                    "true": "Real",
                    "false": "Fake",
                },
            ),
            "pakistan": SourceConfig(
                name="Pakistan_SelfConstructed",
                source_id="Pakistan_SelfConstructed",
                paths=(
                    raw_dir / "pakistan" / "pakistan_dataset.csv",
                    raw_dir / "pakistan" / "pakistan_dataset.parquet",
                    raw_dir / "pakistan" / "pakistan_dataset.jsonl",
                ),
                text_columns=("text", "content", "headline"),
                label_column="label",
                label_mapping={
                    "real": "Real",
                    "fake": "Fake",
                    "satire": "Satire",
                    "true": "Real",
                    "false": "Fake",
                },
                date_columns=("date", "published_at"),
            ),
        }

    def _existing_paths(self, paths: Iterable[Path]) -> List[Path]:
        return [path for path in paths if path.exists()]

    def _read_table(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t", header=None)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in {".jsonl", ".json"}:
            return pd.read_json(path, lines=suffix == ".jsonl")
        raise ValueError(f"Unsupported dataset format: {path}")

    def _coalesce_text(self, df: pd.DataFrame, columns: Tuple[str, ...]) -> pd.Series:
        available = [col for col in columns if col in df.columns]
        if not available:
            raise ValueError(f"No text columns found. Expected one of {columns}, found {list(df.columns)}")
        return (
            df[available]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    def _normalize_labels(self, series: pd.Series, mapping: Dict[str, str]) -> pd.Series:
        normalized = series.astype(str).str.strip().str.lower().map(mapping)
        return normalized

    def _extract_year(self, df: pd.DataFrame, date_columns: Tuple[str, ...]) -> pd.Series:
        for column in date_columns:
            if column in df.columns:
                parsed = pd.to_datetime(df[column], errors="coerce")
                if parsed.notna().any():
                    return parsed.dt.year
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")

    def _load_liar(self) -> pd.DataFrame:
        config = self.source_configs["liar"]
        existing = self._existing_paths(config.paths)
        if len(existing) < 3:
            raise FileNotFoundError(
                "LIAR dataset files not found. Expected train.tsv, valid.tsv, and test.tsv under data/raw/liar/."
            )

        frames = []
        columns = [
            "id",
            "label",
            "statement",
            "subject",
            "speaker",
            "job_title",
            "state",
            "party",
            "barely_true",
            "false_count",
            "half_true",
            "mostly_true",
            "pants_fire",
            "context",
        ]
        for path in existing:
            frame = pd.read_csv(path, sep="\t", header=None)
            frame.columns = columns[: frame.shape[1]]
            frames.append(frame)

        df = pd.concat(frames, ignore_index=True)
        df["text"] = self._coalesce_text(df, config.text_columns)
        df["label"] = self._normalize_labels(df[config.label_column], config.label_mapping)
        df["source_id"] = config.source_id
        df["year"] = 2016
        return df[["text", "label", "source_id", "year"]].dropna(subset=["text", "label"])

    def _load_isot(self) -> pd.DataFrame:
        config = self.source_configs["isot"]
        existing = self._existing_paths(config.paths)
        if len(existing) < 2:
            raise FileNotFoundError(
                "ISOT files not found. Expected Fake.csv and True.csv under data/raw/isot/."
            )

        frames = []
        for path in existing:
            if path.name.lower() not in {"fake.csv", "true.csv"}:
                continue
            frame = pd.read_csv(path)
            frame["label"] = "Fake" if path.name.lower().startswith("fake") else "Real"
            frames.append(frame)

        df = pd.concat(frames, ignore_index=True)
        df["text"] = self._coalesce_text(df, config.text_columns)
        df["label"] = self._normalize_labels(df["label"], config.label_mapping)
        df["source_id"] = config.source_id
        df["year"] = self._extract_year(df, config.date_columns)
        return df[["text", "label", "source_id", "year"]].dropna(subset=["text", "label"])

    def _load_covid(self) -> pd.DataFrame:
        config = self.source_configs["covid"]
        existing = self._existing_paths(config.paths)
        if not existing:
            raise FileNotFoundError(
                "COVID misinformation dataset not found under data/raw/covid/."
            )

        frame = self._read_table(existing[0])
        if config.label_column not in frame.columns:
            alt = next((col for col in ("target", "class") if col in frame.columns), None)
            if alt is None:
                raise ValueError(f"Could not find a label column in {existing[0]}")
            frame[config.label_column] = frame[alt]

        frame["text"] = self._coalesce_text(frame, config.text_columns)
        frame["label"] = self._normalize_labels(frame[config.label_column], config.label_mapping)
        frame["source_id"] = config.source_id
        frame["year"] = 2020
        return frame[["text", "label", "source_id", "year"]].dropna(subset=["text", "label"])

    def _load_pakistan(self) -> pd.DataFrame:
        config = self.source_configs["pakistan"]
        existing = self._existing_paths(config.paths)
        if not existing:
            raise FileNotFoundError(
                "Self-constructed Pakistan dataset not found under data/raw/pakistan/."
            )

        frame = self._read_table(existing[0])
        frame["text"] = self._coalesce_text(frame, config.text_columns)
        frame["label"] = self._normalize_labels(frame[config.label_column], config.label_mapping)
        frame["source_id"] = config.source_id
        frame["year"] = self._extract_year(frame, config.date_columns)
        return frame[["text", "label", "source_id", "year"]].dropna(subset=["text", "label"])

    def load_sources(self, source_names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        source_names = source_names or ["liar", "isot", "covid", "pakistan"]
        loaders = {
            "liar": self._load_liar,
            "isot": self._load_isot,
            "covid": self._load_covid,
            "pakistan": self._load_pakistan,
        }

        datasets: Dict[str, pd.DataFrame] = {}
        for source_name in source_names:
            datasets[source_name] = loaders[source_name]()
            logger.info("Loaded %s with %d rows", source_name, len(datasets[source_name]))
        return datasets

    def combine_datasets(
        self,
        min_samples: int = 5000,
        source_names: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        logger.info("Combining datasets from local source files...")
        datasets = self.load_sources(source_names=source_names)
        combined_df = pd.concat(datasets.values(), ignore_index=True)
        combined_df["text"] = combined_df["text"].fillna("").astype(str).str.strip()
        combined_df = combined_df[combined_df["label"].isin(CANONICAL_CLASSES)]
        combined_df = combined_df[combined_df["text"].str.len() > 0]

        initial_size = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
        duplicate_rate = ((initial_size - len(combined_df)) / initial_size * 100) if initial_size else 0.0

        if len(combined_df) < min_samples:
            raise ValueError(
                f"Combined dataset has {len(combined_df)} samples after deduplication; "
                f"minimum required is {min_samples}. Add more source-backed samples."
            )

        stats = {
            "total_samples": int(len(combined_df)),
            "class_distribution": combined_df["label"].value_counts().to_dict(),
            "duplicate_rate": float(duplicate_rate),
            "source_distribution": combined_df["source_id"].value_counts().to_dict(),
            "avg_text_length": float(combined_df["text"].str.len().mean()),
        }
        return combined_df, stats

    def handle_class_imbalance(
        self,
        df: pd.DataFrame,
        method: str = "undersampling",
    ) -> Tuple[pd.DataFrame, Dict]:
        class_dist = df["label"].value_counts()
        max_pct = float(class_dist.max() / len(df))
        imbalance_detected = max_pct > 0.40
        metadata = {
            "imbalance_detected": imbalance_detected,
            "method": None,
            "class_weights": None,
        }

        if not imbalance_detected:
            self.class_weights_ = {label: 1.0 for label in class_dist.index}
            metadata["class_weights"] = self.class_weights_
            return df, metadata

        if method == "undersampling":
            target_count = int(class_dist.min())
            balanced = (
                df.groupby("label", group_keys=False)
                .apply(lambda part: part.sample(n=target_count, random_state=42))
                .reset_index(drop=True)
            )
            metadata["method"] = "undersampling"
            self.class_weights_ = {label: 1.0 for label in balanced["label"].unique()}
            metadata["class_weights"] = self.class_weights_
            return balanced, metadata

        if method == "class_weighted":
            total = float(len(df))
            num_classes = float(df["label"].nunique())
            self.class_weights_ = {
                label: total / (num_classes * count) for label, count in class_dist.items()
            }
            metadata["method"] = "class_weighted"
            metadata["class_weights"] = self.class_weights_
            return df, metadata

        raise ValueError(
            "SMOTE is not supported for raw text rows in this pipeline. "
            "Use 'undersampling' or 'class_weighted'."
        )

    def generate_reliability_report(self, combined_df: pd.DataFrame, stats: Dict) -> Dict:
        report = {
            "title": "Data Reliability Assessment Report",
            "timestamp": pd.Timestamp.now().isoformat(),
            "dataset_summary": stats,
            "source_scorecards": [],
        }

        for source in combined_df["source_id"].unique():
            source_df = combined_df[combined_df["source_id"] == source]
            scorecard = DataReliabilityScorecard(source)
            report["source_scorecards"].append(
                scorecard.evaluate(
                    source_df,
                    sample_count=int(len(source_df)),
                    year_range=(
                        source_df["year"].dropna().min() if source_df["year"].notna().any() else None,
                        source_df["year"].dropna().max() if source_df["year"].notna().any() else None,
                    ),
                )
            )

        report["overall_reliability_score"] = float(
            np.mean([item["average_score"] for item in report["source_scorecards"]])
        )
        if self.class_weights_ is not None:
            report["class_weight_strategy"] = self.class_weights_
        return report


def main():
    manager = DataSourceManager(data_dir="./data/raw")
    combined_df, stats = manager.combine_datasets(min_samples=5000)
    combined_df, imbalance_meta = manager.handle_class_imbalance(combined_df, method="class_weighted")
    reliability_report = manager.generate_reliability_report(combined_df, stats)
    reliability_report["imbalance_handling"] = imbalance_meta

    output_dir = Path("./data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(output_dir / "combined_dataset.parquet", index=False)
    with open(output_dir / "reliability_report.json", "w", encoding="utf-8") as handle:
        json.dump(reliability_report, handle, indent=2, default=str)

    logger.info("Saved combined dataset with %d rows", len(combined_df))
    return combined_df, reliability_report


if __name__ == "__main__":
    main()
