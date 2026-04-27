"""
PakSentinel: End-to-End NLP Pipeline for Misinformation Detection
Natural Language Processing Assignment 2
"""

__version__ = "1.0.0"
__author__ = "NLP Team"

from .data_sourcing import DataSourceManager
from .data_lake_manager import DataLakeManager
from .nlp_pipeline import NLPPipeline
from .ngram_language_models import LanguageModelClassifier
from .ml_models import MultinomialNaiveBayes, LogisticRegressionClassifier
from .mlflow_integration import MLFlowManager

__all__ = [
    "DataSourceManager",
    "DataLakeManager",
    "NLPPipeline",
    "LanguageModelClassifier",
    "MultinomialNaiveBayes",
    "LogisticRegressionClassifier",
    "MLFlowManager"
]
