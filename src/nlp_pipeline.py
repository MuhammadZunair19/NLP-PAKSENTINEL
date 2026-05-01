"""
Task 3: NLP Processing Pipeline
Real preprocessing analysis for cleaning, tokenization, stopwords,
normalization, and feature representation.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


for resource in ("punkt", "stopwords", "wordnet"):
    try:
        nltk.data.find(f"corpora/{resource}" if resource != "punkt" else "tokenizers/punkt")
    except LookupError:
        nltk.download(resource)


ROMAN_URDU_MARKERS = {
    "acha",
    "awam",
    "baji",
    "banda",
    "bilkul",
    "fake",
    "hai",
    "hain",
    "ho",
    "hukumat",
    "insaaf",
    "jhoot",
    "kia",
    "kya",
    "magar",
    "masla",
    "news",
    "nahi",
    "pakistan",
    "qaum",
    "sach",
    "sarkar",
    "shayad",
    "yeh",
}


class TextCleaner:
    """3.1 Cleaning utilities."""

    def __init__(self):
        self.patterns = {
            "html_tags": r"<[^>]+>",
            "urls": r"http[s]?://\S+|www\.\S+",
            "emails": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "mentions": r"@\w+",
            "hashtags": r"#(\w+)",
            "repeated_punctuation": r"([!?.,])\1{1,}",
            "roman_urdu_codeswitch": r"\b(yaar|acha|nahi|kya|hai|hain|magar|lekin|awam|hukumat)\b",
        }
        self.emoji_pattern = re.compile(
            "["
            "\U0001F300-\U0001FAFF"
            "\U00002700-\U000027BF"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = re.sub(self.patterns["html_tags"], " ", text)
        text = re.sub(self.patterns["urls"], " [URL] ", text)
        text = re.sub(self.patterns["emails"], " [EMAIL] ", text)
        text = re.sub(self.patterns["mentions"], " [MENTION] ", text)
        text = re.sub(self.patterns["hashtags"], r" \1 ", text)
        text = re.sub(self.patterns["repeated_punctuation"], r"\1", text)
        text = self.emoji_pattern.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    def audit_cleaning(self, df: pd.DataFrame, text_col: str = "text", sample_size: int = 200) -> Dict:
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42).copy()

        rows = []
        total_removed = 0
        html_hits = 0
        url_hits = 0
        mention_hits = 0
        emoji_hits = 0
        repeated_punct_hits = 0

        for idx, original in sample_df[text_col].fillna("").astype(str).items():
            cleaned = self.clean(original)
            total_removed += max(len(original) - len(cleaned), 0)
            html_hits += int(bool(re.search(self.patterns["html_tags"], original)))
            url_hits += int(bool(re.search(self.patterns["urls"], original)))
            mention_hits += int(bool(re.search(self.patterns["mentions"], original)))
            emoji_hits += int(bool(self.emoji_pattern.search(original)))
            repeated_punct_hits += int(bool(re.search(self.patterns["repeated_punctuation"], original)))
            rows.append(
                {
                    "index": idx,
                    "before": original[:200],
                    "after": cleaned[:200],
                    "before_len": len(original),
                    "after_len": len(cleaned),
                }
            )

        metrics = {
            "sample_size": len(rows),
            "avg_original_length": float(np.mean([row["before_len"] for row in rows])) if rows else 0.0,
            "avg_cleaned_length": float(np.mean([row["after_len"] for row in rows])) if rows else 0.0,
            "char_reduction_total": int(total_removed),
            "html_records": html_hits,
            "url_records": url_hits,
            "mention_records": mention_hits,
            "emoji_records": emoji_hits,
            "repeated_punctuation_records": repeated_punct_hits,
        }
        return {"sample_rows": rows, "metrics": metrics}


class TokenizerComparison:
    """3.2 Compare tokenizers with measurable outputs."""

    def __init__(self):
        self.nltk_tokenizer = word_tokenize
        try:
            self.spacy_nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.spacy_nlp = spacy.blank("en")
        self.regex_tokenizer = RegexpTokenizer(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|(?:\[URL\]|\[EMAIL\]|\[MENTION\])")

    def tokenize_nltk(self, text: str) -> List[str]:
        return self.nltk_tokenizer(text)

    def tokenize_spacy(self, text: str) -> List[str]:
        return [token.text for token in self.spacy_nlp(text)]

    def tokenize_regex(self, text: str) -> List[str]:
        return self.regex_tokenizer.tokenize(text)

    def _evaluate_tokenizer(
        self,
        tokenizer_name: str,
        tokenizer,
        sample_texts: Sequence[str],
        reference_vocab: Set[str],
    ) -> Dict:
        token_counts: List[int] = []
        elapsed: List[float] = []
        oov_total = 0
        total_tokens = 0
        contractions_kept = 0
        contraction_cases = 0
        roman_kept = 0
        roman_cases = 0

        for text in sample_texts:
            start = time.perf_counter()
            tokens = tokenizer(text)
            elapsed.append(time.perf_counter() - start)
            token_counts.append(len(tokens))

            lowered_tokens = [token.lower() for token in tokens]
            total_tokens += len(lowered_tokens)
            oov_total += sum(1 for token in lowered_tokens if token.isalpha() and token not in reference_vocab)

            original_words = re.findall(r"\b\w+'\w+\b", text.lower())
            contraction_cases += len(original_words)
            contractions_kept += sum(1 for word in original_words if word in lowered_tokens)

            roman_in_text = [word for word in re.findall(r"[a-z]+", text.lower()) if word in ROMAN_URDU_MARKERS]
            roman_cases += len(roman_in_text)
            roman_kept += sum(1 for word in roman_in_text if word in lowered_tokens)

        return {
            "tokenizer": tokenizer_name,
            "avg_tokens_per_doc": float(np.mean(token_counts)) if token_counts else 0.0,
            "avg_processing_time_ms": float(np.mean(elapsed) * 1000) if elapsed else 0.0,
            "oov_rate": float(oov_total / total_tokens) if total_tokens else 0.0,
            "contraction_handling_rate": float(contractions_kept / contraction_cases) if contraction_cases else 1.0,
            "roman_urdu_handling_rate": float(roman_kept / roman_cases) if roman_cases else 1.0,
            "sample_size": len(sample_texts),
        }

    def compare_tokenizers(self, df: pd.DataFrame, text_col: str = "text", sample_size: int = 50) -> Dict:
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        sample_texts = sample_df[text_col].fillna("").astype(str).tolist()

        reference_counter = Counter(
            token.lower()
            for text in df[text_col].fillna("").astype(str).tolist()
            for token in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
        )
        reference_vocab = {token for token, count in reference_counter.items() if count >= 2}

        results = {
            "nltk": self._evaluate_tokenizer("nltk", self.tokenize_nltk, sample_texts, reference_vocab),
            "spacy": self._evaluate_tokenizer("spacy", self.tokenize_spacy, sample_texts, reference_vocab),
            "regex": self._evaluate_tokenizer("regex", self.tokenize_regex, sample_texts, reference_vocab),
        }

        scored = sorted(
            results.items(),
            key=lambda item: (
                item[1]["oov_rate"],
                -item[1]["contraction_handling_rate"],
                -item[1]["roman_urdu_handling_rate"],
                item[1]["avg_processing_time_ms"],
            ),
        )
        results["recommendation"] = scored[0][0]
        return results


class StopwordManager:
    """3.3 Stopword experiments."""

    def __init__(self):
        self.default_stopwords = set(stopwords.words("english"))
        self.custom_stopwords = self._create_domain_stopwords()

    def _create_domain_stopwords(self) -> Set[str]:
        keep_words = {
            "not",
            "no",
            "nor",
            "never",
            "cannot",
            "can't",
            "won't",
            "don't",
            "shouldn't",
            "completely",
            "absolutely",
            "definitely",
            "clearly",
            "obviously",
            "without",
        }
        custom = set(self.default_stopwords) - keep_words
        custom.update(
            {
                "breaking",
                "share",
                "post",
                "article",
                "news",
                "reported",
                "reportedly",
                "says",
                "said",
                "via",
                "amp",
                "thread",
                "update",
                "watch",
                "read",
            }
        )
        return custom

    def remove_stopwords(self, tokens: List[str], use_custom: bool = True) -> Tuple[List[str], List[str]]:
        active = self.custom_stopwords if use_custom else self.default_stopwords
        kept, removed = [], []
        for token in tokens:
            if token.lower() in active:
                removed.append(token)
            else:
                kept.append(token)
        return kept, removed

    def _classification_f1(self, texts: Sequence[str], labels: Sequence[str]) -> float:
        X_train, X_test, y_train, y_test = train_test_split(
            list(texts),
            list(labels),
            test_size=0.2,
            random_state=42,
            stratify=list(labels),
        )
        vectorizer = TfidfVectorizer(max_features=3000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        model = LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="auto")
        model.fit(X_train_vec, y_train)
        predictions = model.predict(X_test_vec)
        return float(f1_score(y_test, predictions, average="weighted", zero_division=0))

    def analyze_stopword_impact(
        self,
        df: pd.DataFrame,
        tokenizer,
        text_col: str = "text",
        label_col: str = "label",
    ) -> Dict:
        sample_df = df.sample(n=min(len(df), 1200), random_state=42).copy()

        default_removed = []
        custom_removed = []
        default_docs = []
        custom_docs = []
        total_tokens = 0

        for text in sample_df[text_col].fillna("").astype(str):
            tokens = tokenizer(text)
            total_tokens += len(tokens)
            default_kept, default_drop = self.remove_stopwords(tokens, use_custom=False)
            custom_kept, custom_drop = self.remove_stopwords(tokens, use_custom=True)
            default_removed.extend(default_drop)
            custom_removed.extend(custom_drop)
            default_docs.append(" ".join(default_kept))
            custom_docs.append(" ".join(custom_kept))

        default_f1 = self._classification_f1(default_docs, sample_df[label_col].tolist())
        custom_f1 = self._classification_f1(custom_docs, sample_df[label_col].tolist())

        return {
            "default_stopwords": {
                "token_removal_rate": float(len(default_removed) / total_tokens) if total_tokens else 0.0,
                "unique_removed": sorted(set(default_removed))[:50],
                "weighted_f1": default_f1,
            },
            "custom_stopwords": {
                "token_removal_rate": float(len(custom_removed) / total_tokens) if total_tokens else 0.0,
                "unique_removed": sorted(set(custom_removed))[:50],
                "weighted_f1": custom_f1,
            },
            "f1_delta_custom_minus_default": float(custom_f1 - default_f1),
            "justified_modifications": {
                "kept_from_default": sorted(
                    {"not", "no", "nor", "never", "cannot", "can't", "won't", "don't", "shouldn't", "completely", "absolutely", "definitely", "clearly", "obviously", "without"}
                ),
                "added_domain_stopwords": sorted(
                    {"breaking", "share", "post", "article", "news", "reported", "reportedly", "says", "said", "via", "amp", "thread", "update", "watch", "read"}
                ),
            },
        }


class NormalizationModule:
    """3.4 Stemming vs lemmatization."""

    def __init__(self):
        self.porter = PorterStemmer()
        self.snowball = SnowballStemmer("english")
        self.wordnet = WordNetLemmatizer()

    def stem_porter(self, tokens: List[str]) -> List[str]:
        return [self.porter.stem(token) for token in tokens]

    def stem_snowball(self, tokens: List[str]) -> List[str]:
        return [self.snowball.stem(token) for token in tokens]

    def lemmatize_wordnet(self, tokens: List[str]) -> List[str]:
        return [self.wordnet.lemmatize(token) for token in tokens]

    def compare_normalization(self, test_terms: Sequence[str] | None = None) -> Dict:
        if test_terms is None:
            test_terms = [
                "misinformation",
                "misleading",
                "fabricated",
                "verification",
                "veracity",
                "credibility",
                "elections",
                "vaccinations",
                "headlines",
                "rumours",
                "conspiracies",
                "broadcasting",
                "journalists",
                "factchecking",
                "communities",
                "sharing",
                "viral",
                "governments",
                "authorities",
                "disinformation",
            ]

        methods = {
            "porter": self.stem_porter,
            "snowball": self.stem_snowball,
            "wordnet": self.lemmatize_wordnet,
        }
        outputs: Dict[str, List[str]] = {}
        timings: Dict[str, float] = {}
        overstemming: Dict[str, List[Dict[str, str]]] = {}

        for name, method in methods.items():
            start = time.perf_counter()
            normalized = method(list(test_terms))
            timings[name] = (time.perf_counter() - start) * 1000
            outputs[name] = normalized
            overstemming[name] = [
                {"original": original, "normalized": norm}
                for original, norm in zip(test_terms, normalized)
                if name != "wordnet" and len(norm) <= max(2, int(len(original) * 0.5))
            ]

        vocab_reduction = {
            name: float(1 - (len(set(values)) / len(test_terms))) for name, values in outputs.items()
        }

        return {
            "terms": list(test_terms),
            "normalized_terms": outputs,
            "processing_time_ms": timings,
            "vocabulary_reduction": vocab_reduction,
            "over_stemming_examples": overstemming,
            "recommendation": "wordnet",
        }


class FeatureRepresentation:
    """3.5 Feature representation experiments."""

    def __init__(self):
        self.bow_vectorizer: CountVectorizer | None = None
        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.word2vec_model: Word2Vec | None = None

    def create_bow_features(self, texts: Sequence[str], max_features: int = 5000) -> Tuple[np.ndarray, Dict]:
        self.bow_vectorizer = CountVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b")
        matrix = self.bow_vectorizer.fit_transform(texts).toarray()
        stats = {
            "shape": list(matrix.shape),
            "sparsity": float(1.0 - (np.count_nonzero(matrix) / matrix.size)) if matrix.size else 0.0,
            "vocabulary_size": len(self.bow_vectorizer.vocabulary_),
        }
        return matrix, stats

    def get_top_bow_terms(self, texts: Sequence[str], labels: Sequence[str], top_n: int = 30) -> Dict:
        if self.bow_vectorizer is None:
            self.create_bow_features(texts)
        matrix = self.bow_vectorizer.transform(texts)
        vocab = np.array(self.bow_vectorizer.get_feature_names_out())
        label_series = pd.Series(labels)
        results = {}
        for label in sorted(label_series.unique()):
            class_matrix = matrix[label_series == label]
            feature_sums = np.asarray(class_matrix.sum(axis=0)).ravel()
            top_indices = np.argsort(feature_sums)[-top_n:][::-1]
            results[label] = [(vocab[idx], float(feature_sums[idx])) for idx in top_indices]
        return results

    def create_tfidf_features(
        self,
        texts: Sequence[str],
        variant: str = "standard",
        max_features: int = 5000,
    ) -> Tuple[np.ndarray, TfidfVectorizer, Dict]:
        kwargs = {"max_features": max_features}
        if variant == "smooth":
            kwargs.update({"smooth_idf": True, "sublinear_tf": False})
        elif variant == "sublinear":
            kwargs.update({"smooth_idf": True, "sublinear_tf": True})
        else:
            kwargs.update({"smooth_idf": False, "sublinear_tf": False})

        self.tfidf_vectorizer = TfidfVectorizer(**kwargs)
        matrix = self.tfidf_vectorizer.fit_transform(texts)
        dense = matrix.toarray()
        stats = {
            "shape": list(dense.shape),
            "sparsity": float(1.0 - (np.count_nonzero(dense) / dense.size)) if dense.size else 0.0,
            "vocabulary_size": len(self.tfidf_vectorizer.vocabulary_),
        }
        return dense, self.tfidf_vectorizer, stats

    def get_top_tfidf_terms_per_class(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        variant: str,
        top_n: int = 15,
    ) -> Dict[str, List[Tuple[str, float]]]:
        matrix, vectorizer, _ = self.create_tfidf_features(texts, variant=variant)
        feature_names = np.array(vectorizer.get_feature_names_out())
        label_series = pd.Series(labels)
        results: Dict[str, List[Tuple[str, float]]] = {}
        for label in sorted(label_series.unique()):
            class_rows = matrix[label_series == label]
            class_mean = class_rows.mean(axis=0)
            other_mean = matrix[label_series != label].mean(axis=0)
            scores = class_mean - other_mean
            top_indices = np.argsort(scores)[-top_n:][::-1]
            results[label] = [(feature_names[idx], float(scores[idx])) for idx in top_indices]
        return results

    def cosine_similarity_retrieval(
        self,
        query_text: str,
        corpus: Sequence[str],
        tfidf_matrix: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer has not been fitted.")
        query_vec = self.tfidf_vectorizer.transform([query_text]).toarray()
        similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
        indices = np.argsort(similarities)[-top_k:][::-1]
        return [(corpus[idx], float(similarities[idx])) for idx in indices]

    def train_word2vec(
        self,
        tokenized_texts: Sequence[Sequence[str]],
        model_type: str = "skip-gram",
        vector_size: int = 200,
        window: int = 5,
        min_count: int = 3,
    ) -> Tuple[Word2Vec, Dict]:
        self.word2vec_model = Word2Vec(
            sentences=[list(tokens) for tokens in tokenized_texts],
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=1 if model_type == "skip-gram" else 0,
            workers=1,
            seed=42,
        )
        return self.word2vec_model, {
            "model_type": model_type,
            "vector_size": vector_size,
            "window": window,
            "min_count": min_count,
            "vocabulary_size": len(self.word2vec_model.wv),
        }

    def get_word_similarities(self, word: str, top_n: int = 5) -> List[Tuple[str, float]]:
        if self.word2vec_model is None or word not in self.word2vec_model.wv:
            return []
        return [(neighbor, float(score)) for neighbor, score in self.word2vec_model.wv.most_similar(word, topn=top_n)]

    def visualize_embeddings(self, save_path: str, max_words: int = 100) -> str:
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model has not been trained.")
        words = self.word2vec_model.wv.index_to_key[:max_words]
        vectors = np.array([self.word2vec_model.wv[word] for word in words])
        perplexity = max(5, min(30, len(words) - 1))
        projection = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(vectors)

        plt.figure(figsize=(14, 10))
        plt.scatter(projection[:, 0], projection[:, 1], s=16, alpha=0.7)
        for idx, word in enumerate(words):
            plt.annotate(word, (projection[idx, 0], projection[idx, 1]), fontsize=7)
        plt.title("Word2Vec t-SNE")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        return save_path

    def _document_vectors(self, tokenized_texts: Sequence[Sequence[str]]) -> np.ndarray:
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model has not been trained.")
        dim = self.word2vec_model.vector_size
        vectors = []
        for tokens in tokenized_texts:
            valid = [self.word2vec_model.wv[token] for token in tokens if token in self.word2vec_model.wv]
            vectors.append(np.mean(valid, axis=0) if valid else np.zeros(dim))
        return np.vstack(vectors)

    def compare_feature_families(
        self,
        texts: Sequence[str],
        tokenized_texts: Sequence[Sequence[str]],
        labels: Sequence[str],
    ) -> Dict:
        X_train_text, X_test_text, X_train_tok, X_test_tok, y_train, y_test = train_test_split(
            list(texts),
            [list(tokens) for tokens in tokenized_texts],
            list(labels),
            test_size=0.2,
            random_state=42,
            stratify=list(labels),
        )

        tfidf = TfidfVectorizer(max_features=4000)
        X_train_tfidf = tfidf.fit_transform(X_train_text)
        X_test_tfidf = tfidf.transform(X_test_text)

        tfidf_model = LogisticRegression(max_iter=1000, class_weight="balanced")
        tfidf_model.fit(X_train_tfidf, y_train)
        tfidf_pred = tfidf_model.predict(X_test_tfidf)

        word2vec_only = LogisticRegression(max_iter=1000, class_weight="balanced")
        X_train_w2v = self._document_vectors(X_train_tok)
        X_test_w2v = self._document_vectors(X_test_tok)
        word2vec_only.fit(X_train_w2v, y_train)
        w2v_pred = word2vec_only.predict(X_test_w2v)

        concat_model = LogisticRegression(max_iter=1000, class_weight="balanced")
        X_train_concat = np.hstack([X_train_tfidf.toarray(), X_train_w2v])
        X_test_concat = np.hstack([X_test_tfidf.toarray(), X_test_w2v])
        concat_model.fit(X_train_concat, y_train)
        concat_pred = concat_model.predict(X_test_concat)

        return {
            "tfidf_only_f1": float(f1_score(y_test, tfidf_pred, average="weighted", zero_division=0)),
            "word2vec_only_f1": float(f1_score(y_test, w2v_pred, average="weighted", zero_division=0)),
            "concatenated_f1": float(f1_score(y_test, concat_pred, average="weighted", zero_division=0)),
        }


class NLPPipeline:
    """Run the assignment-oriented NLP pipeline."""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cleaner = TextCleaner()
        self.tokenizer_comparison = TokenizerComparison()
        self.stopword_manager = StopwordManager()
        self.normalization = NormalizationModule()
        self.feature_extraction = FeatureRepresentation()

    def _tokenize_recommended(self, text: str) -> List[str]:
        return self.tokenizer_comparison.tokenize_nltk(text)

    def process_dataset(self, df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> Tuple[Dict, pd.DataFrame]:
        df = df.copy()
        results: Dict[str, object] = {}

        results["cleaning_audit"] = self.cleaner.audit_cleaning(df, text_col=text_col, sample_size=200)
        df["cleaned_text"] = df[text_col].fillna("").astype(str).map(self.cleaner.clean)

        results["tokenizer_comparison"] = self.tokenizer_comparison.compare_tokenizers(
            df, text_col="cleaned_text", sample_size=50
        )
        df["tokens"] = df["cleaned_text"].map(self._tokenize_recommended)

        results["stopword_analysis"] = self.stopword_manager.analyze_stopword_impact(
            df, self._tokenize_recommended, text_col="cleaned_text", label_col=label_col
        )
        df["tokens_no_stopwords"] = df["tokens"].map(lambda tokens: self.stopword_manager.remove_stopwords(tokens, True)[0])

        results["normalization_comparison"] = self.normalization.compare_normalization()

        bow_matrix, bow_stats = self.feature_extraction.create_bow_features(df["cleaned_text"].tolist())
        results["bow"] = {
            "stats": bow_stats,
            "top_terms_per_class": self.feature_extraction.get_top_bow_terms(
                df["cleaned_text"].tolist(),
                df[label_col].tolist(),
            ),
            "why_bow_is_insufficient": (
                "BoW collapses order and local context, so it cannot distinguish phrase-level "
                "patterns such as negation, hedging, or claim framing that matter in misinformation."
            ),
        }

        tfidf_results = {}
        retrieval_examples = []
        for variant in ("standard", "smooth", "sublinear"):
            tfidf_matrix, _, stats = self.feature_extraction.create_tfidf_features(df["cleaned_text"].tolist(), variant=variant)
            tfidf_results[variant] = {
                "stats": stats,
                "top_terms_per_class": self.feature_extraction.get_top_tfidf_terms_per_class(
                    df["cleaned_text"].tolist(),
                    df[label_col].tolist(),
                    variant=variant,
                ),
            }
            if variant == "sublinear":
                sample_queries = df["cleaned_text"].head(min(10, len(df))).tolist()
                retrieval_examples = [
                    {
                        "query": query,
                        "results": self.feature_extraction.cosine_similarity_retrieval(
                            query,
                            df["cleaned_text"].tolist(),
                            tfidf_matrix,
                            top_k=3,
                        ),
                    }
                    for query in sample_queries
                ]
        results["tfidf"] = tfidf_results
        results["retrieval_examples"] = retrieval_examples

        tokenized_texts = df["tokens_no_stopwords"].tolist()
        similarity_pairs = [("government", "policy"), ("fake", "viral"), ("health", "vaccine"), ("court", "verdict")]
        word2vec_results = {}
        for model_type in ("cbow", "skip-gram"):
            model, stats = self.feature_extraction.train_word2vec(tokenized_texts, model_type=model_type)
            pair_scores = {}
            for left, right in similarity_pairs:
                if left in model.wv and right in model.wv:
                    pair_scores[f"{left}::{right}"] = float(model.wv.similarity(left, right))
            key_terms = {}
            for term in ("government", "fake", "health", "election"):
                key_terms[term] = self.feature_extraction.get_word_similarities(term, top_n=5)
            word2vec_results[model_type] = {
                "stats": stats,
                "pair_similarities": pair_scores,
                "neighbors": key_terms,
            }

        tsne_path = str(self.output_dir / "word2vec_tsne.png")
        results["word2vec"] = {
            "models": word2vec_results,
            "tsne_path": self.feature_extraction.visualize_embeddings(tsne_path),
            "feature_family_f1": self.feature_extraction.compare_feature_families(
                df["cleaned_text"].tolist(),
                tokenized_texts,
                df[label_col].tolist(),
            ),
        }

        return results, df


def main():
    sample_path = Path("./data/raw/combined_dataset.parquet")
    if not sample_path.exists():
        raise FileNotFoundError("Run Task 1 first so ./data/raw/combined_dataset.parquet exists.")

    df = pd.read_parquet(sample_path)
    pipeline = NLPPipeline(output_dir="./output")
    results, processed_df = pipeline.process_dataset(df, text_col="text", label_col="label")

    with open("./output/task3_pipeline_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    processed_df.to_parquet("./output/task3_processed_data.parquet", index=False)
    logger.info("Saved Task 3 outputs to ./output")
    return processed_df, results


if __name__ == "__main__":
    main()
