"""
Task 4: N-Gram Language Models with Kneser-Ney Smoothing
Builds unigram, bigram, and trigram models for fake/real news classification.

Justification for Kneser-Ney Smoothing:
- Laplace smoothing adds too much probability mass to unseen events
- Kneser-Ney considers continuation probability (how often a word appears after different words)
- Better handles rare word combinations in misinformation datasets
- Particularly effective for small n-gram datasets
"""

import numpy as np
import pandas as pd
import logging
import json
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NGramModel:
    """Base N-Gram Language Model"""
    
    def __init__(self, n: int, smoothing: str = 'laplace', smoothing_param: float = 1.0):
        """
        Initialize N-Gram model
        
        Args:
            n: N-gram order (1, 2, or 3)
            smoothing: Smoothing method ('laplace', 'kneser-ney')
            smoothing_param: Smoothing parameter
        """
        self.n = n
        self.smoothing = smoothing
        self.smoothing_param = smoothing_param
        
        self.ngrams = defaultdict(int)  # Count n-grams
        self.context_count = defaultdict(int)  # Count (n-1)-gram contexts
        self.unigrams = defaultdict(int)  # Unigram counts
        
        self.vocab_size = 0
        self.total_ngrams = 0
    
    def train(self, texts: List[str], tokenized: bool = False) -> None:
        """
        Train language model on texts
        
        Args:
            texts: List of documents
            tokenized: Whether texts are already tokenized
        """
        
        logger.info(f"Training {self.n}-gram model with {self.smoothing} smoothing...")
        
        for text in texts:
            if not tokenized:
                tokens = text.lower().split()
            else:
                tokens = text
            
            # Add start tokens
            tokens = ['<START>'] * (self.n - 1) + tokens + ['<END>']
            
            # Extract n-grams
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i+self.n])
                context = ngram[:-1]
                token = ngram[-1]
                
                self.ngrams[ngram] += 1
                self.context_count[context] += 1
                self.unigrams[token] += 1
                self.total_ngrams += 1
        
        self.vocab_size = len(self.unigrams)
        
        logger.info(f"Trained {self.n}-gram model: vocab_size={self.vocab_size}, "
                   f"unique_ngrams={len(self.ngrams)}, total_ngrams={self.total_ngrams}")
    
    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Get probability of n-gram
        
        Args:
            ngram: Tuple of tokens
        
        Returns:
            Probability
        """
        
        if len(ngram) != self.n:
            raise ValueError(f"Expected {self.n}-gram, got {len(ngram)}-gram")
        
        if self.smoothing == 'laplace':
            return self._laplace_probability(ngram)
        elif self.smoothing == 'kneser-ney':
            return self._kneser_ney_probability(ngram)
        else:
            raise ValueError(f"Unknown smoothing method: {self.smoothing}")
    
    def _laplace_probability(self, ngram: Tuple[str, ...]) -> float:
        """Laplace (add-one) smoothing"""
        
        context = ngram[:-1]
        count = self.ngrams[ngram]
        context_total = self.context_count[context]
        
        # P(w|context) = (count + k) / (context_total + k*V)
        numerator = count + self.smoothing_param
        denominator = context_total + self.smoothing_param * self.vocab_size
        
        if denominator == 0:
            return 1.0 / self.vocab_size
        
        return numerator / denominator
    
    def _kneser_ney_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Kneser-Ney smoothing
        
        Formula: P_KN(w|context) = max(0, count(context,w) - d) / count(context)
                                    + (d/count(context)) * N_1(*, context) * P_KN(w)
        
        Where:
        - d: discount parameter (0 < d < 1)
        - N_1(*, context): number of unique continuations after context
        """
        
        context = ngram[:-1]
        token = ngram[-1]
        count = self.ngrams[ngram]
        context_total = self.context_count[context]
        
        if context_total == 0:
            return 1.0 / self.vocab_size
        
        # Discount parameter (empirically optimal around 0.75)
        d = 0.75
        
        # First term: discounted probability
        numerator = max(0, count - d)
        first_term = numerator / context_total if context_total > 0 else 0
        
        # Second term: backoff probability
        if len(context) > 0:
            # Count unique contexts that end with this token
            unique_contexts = len([ng for ng in self.ngrams if ng[-1] == token])
            
            # Normalize by total unique (w) in the corpus
            total_unique = len(set(ngram[-1] for ngram in self.ngrams))
            
            backoff_weight = (d / context_total) * unique_contexts if context_total > 0 else 0
            backoff_prob = self._kneser_ney_probability(context + (token,)) if len(context) < self.n - 1 else self.unigrams[token] / self.total_ngrams
            
            second_term = backoff_weight * backoff_prob
        else:
            # Unigram probability
            second_term = 0
        
        return first_term + second_term
    
    def perplexity(self, texts: List[str], tokenized: bool = False) -> float:
        """
        Calculate perplexity on test data
        
        Args:
            texts: List of test documents
            tokenized: Whether texts are already tokenized
        
        Returns:
            Perplexity score (lower is better)
        """
        
        total_log_prob = 0
        total_tokens = 0
        
        for text in texts:
            if not tokenized:
                tokens = text.lower().split()
            else:
                tokens = text
            
            tokens = ['<START>'] * (self.n - 1) + tokens + ['<END>']
            
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i+self.n])
                prob = self.get_probability(ngram)
                
                if prob > 0:
                    total_log_prob += math.log(prob)
                else:
                    total_log_prob += math.log(1e-10)  # Avoid log(0)
                
                total_tokens += 1
        
        if total_tokens == 0:
            return float('inf')
        
        # Perplexity = exp(-1/N * sum(log(p)))
        perplexity = math.exp(-total_log_prob / total_tokens)
        
        return perplexity
    
    def get_top_ngrams(self, top_n: int = 20) -> List[Tuple]:
        """Get top n-grams by frequency"""
        
        sorted_ngrams = sorted(self.ngrams.items(), key=lambda x: x[1], reverse=True)
        return [(ng, count) for ng, count in sorted_ngrams[:top_n]]


class LanguageModelClassifier:
    """
    Classify documents using language models
    
    Method: Calculate perplexity of document under each class LM
    Assign to class with lowest perplexity
    """
    
    def __init__(self, n: int = 3, smoothing: str = 'kneser-ney'):
        """Initialize classifier"""
        self.n = n
        self.smoothing = smoothing
        self.models = {}
        self.classes = None
    
    def train(self, texts: List[str], labels: List[str]) -> Dict:
        """Train separate language models for each class"""
        
        logger.info(f"Training language model classifier...")
        
        self.classes = list(set(labels))
        stats = {}
        
        for label in self.classes:
            class_texts = [texts[i] for i, l in enumerate(labels) if l == label]
            
            model = NGramModel(n=self.n, smoothing=self.smoothing)
            model.train(class_texts)
            self.models[label] = model
            
            stats[label] = {
                "vocab_size": model.vocab_size,
                "unique_ngrams": len(model.ngrams),
                "training_documents": len(class_texts),
                "top_ngrams": model.get_top_ngrams(5)
            }
        
        logger.info(f"Trained {len(self.classes)} language models")
        
        return stats
    
    def predict(self, text: str) -> Tuple[str, Dict]:
        """
        Classify document using language model perplexity
        
        Args:
            text: Document to classify
        
        Returns:
            Tuple of (predicted_class, perplexity_scores)
        """
        
        tokens = text.lower().split()
        perplexities = {}
        
        for label, model in self.models.items():
            perp = model.perplexity([tokens], tokenized=True)
            perplexities[label] = perp
        
        # Predict class with lowest perplexity
        predicted_class = min(perplexities, key=perplexities.get)
        
        return predicted_class, perplexities
    
    def evaluate(self, texts: List[str], labels: List[str]) -> Dict:
        """Evaluate classifier on test set"""
        
        logger.info("Evaluating language model classifier...")
        
        predictions = []
        perplexities_list = []
        
        for text, true_label in zip(texts, labels):
            pred_label, perps = self.predict(text)
            predictions.append(pred_label)
            perplexities_list.append(perps)
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import confusion_matrix, classification_report
        
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
            "classification_report": classification_report(labels, predictions, output_dict=True)
        }
        
        logger.info(f"Results: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        return results


class NGramAnalyzer:
    """Analyze n-gram patterns in dataset"""
    
    @staticmethod
    def extract_ngrams(texts: List[str], n: int,
                       min_count: int = 1) -> Dict[Tuple, int]:
        """Extract n-grams from texts"""
        
        ngrams = defaultdict(int)
        
        for text in texts:
            tokens = text.lower().split()
            tokens = ['<START>'] * (n - 1) + tokens + ['<END>']
            
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                ngrams[ngram] += 1
        
        # Filter by minimum count
        ngrams = {ng: count for ng, count in ngrams.items() if count >= min_count}
        
        return dict(ngrams)
    
    @staticmethod
    def analyze_dataset(texts: List[str], labels: List[str]) -> Dict:
        """Analyze n-gram patterns per class"""
        
        logger.info("Analyzing n-gram patterns...")
        
        analysis = {}
        
        for label in set(labels):
            label_texts = [texts[i] for i, l in enumerate(labels) if l == label]
            
            analysis[label] = {
                "unigrams": NGramAnalyzer.extract_ngrams(label_texts, 1, min_count=2),
                "bigrams": NGramAnalyzer.extract_ngrams(label_texts, 2, min_count=2),
                "trigrams": NGramAnalyzer.extract_ngrams(label_texts, 3, min_count=2)
            }
        
        return analysis
    
    @staticmethod
    def compare_ngrams(analysis: Dict) -> Dict:
        """Compare discriminative n-grams between classes"""
        
        logger.info("Comparing n-grams between classes...")
        
        comparison = {}
        
        for label in analysis:
            for n_type in ['unigrams', 'bigrams', 'trigrams']:
                ngrams = analysis[label][n_type]
                top_ngrams = sorted(ngrams.items(), key=lambda x: x[1], reverse=True)[:20]
                
                key = f"{label}_{n_type}"
                comparison[key] = top_ngrams
        
        return comparison


def main():
    """Run n-gram language model pipeline"""
    
    # Sample training data
    real_texts = [
        "Government announces new education policy",
        "Stock market rises by 5 percent",
        "New university admissions open",
        "Infrastructure project completed"
    ]
    
    fake_texts = [
        "Celebrity dies in car crash false alarm",
        "Vaccine causes autism hoax spreads",
        "Hidden government conspiracy revealed",
        "Million dollar money making scheme"
    ]
    
    test_texts = [
        "Official statement from ministry",
        "Unconfirmed viral report spreads",
        "Breaking news about elections"
    ]
    
    test_labels = ["Real", "Fake", "Real"]
    
    # Create datasets
    train_texts = real_texts + fake_texts
    train_labels = ["Real"] * len(real_texts) + ["Fake"] * len(fake_texts)
    
    # Train language model classifier
    lm_classifier = LanguageModelClassifier(n=3, smoothing='kneser-ney')
    lm_stats = lm_classifier.train(train_texts, train_labels)
    
    # Evaluate
    eval_results = lm_classifier.evaluate(test_texts, test_labels)
    
    # Analyze n-grams
    analysis = NGramAnalyzer.analyze_dataset(train_texts, train_labels)
    comparison = NGramAnalyzer.compare_ngrams(analysis)
    
    # Save results
    results = {
        "language_model_stats": lm_stats,
        "evaluation": eval_results,
        "ngram_comparison": {k: [(ng, count) for ng, count in v] 
                            for k, v in comparison.items()}
    }
    
    with open('./ngram_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("N-gram pipeline completed!")
    
    return results


if __name__ == "__main__":
    results = main()
