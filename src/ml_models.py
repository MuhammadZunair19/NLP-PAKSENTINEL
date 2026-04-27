"""
Task 5: Machine Learning Models for Misinformation Detection
5.1: Custom Multinomial Naive Bayes (no sklearn)
5.2: Logistic Regression with various regularization
5.3: Polynomial Features + LR with PCA
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, roc_curve, auc,
                            classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== TASK 5.1: CUSTOM NAIVE BAYES ==========

class MultinomialNaiveBayes:
    """
    Custom Multinomial Naive Bayes implementation from scratch
    
    Why no sklearn:
    - Educational transparency: understand the algorithm
    - Full control over implementation details
    - Custom logging for misclassifications
    - Detailed probability tracking
    
    Mathematical formulation:
    P(class|features) = P(features|class) * P(class) / P(features)
    
    With Laplace smoothing:
    P(word_i|class) = (count(word_i, class) + alpha) / (sum_words_in_class + alpha * vocab_size)
    """
    
    def __init__(self, alpha: float = 1.0, log_space: bool = True):
        """
        Initialize Naive Bayes
        
        Args:
            alpha: Laplace smoothing parameter
            log_space: Use log-space to avoid underflow
        """
        self.alpha = alpha
        self.log_space = log_space
        
        self.class_counts = {}
        self.feature_counts = {}
        self.vocab_size = 0
        self.classes = None
        self.total_docs = 0
        self.feature_names = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """
        Train Naive Bayes classifier
        
        Args:
            X: Feature matrix (dense or sparse)
            y: Class labels
            feature_names: Feature names for interpretation
        """
        
        logger.info("Training Multinomial Naive Bayes...")
        
        if hasattr(X, 'toarray'):  # Sparse matrix
            X = X.toarray()
        
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        self.total_docs = len(y)
        self.vocab_size = X.shape[1]
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self.vocab_size)]
        
        # Initialize counts
        for cls in self.classes:
            self.class_counts[cls] = 0
            self.feature_counts[cls] = np.zeros(self.vocab_size)
        
        # Calculate counts
        for i, (features, label) in enumerate(zip(X, y)):
            self.class_counts[label] += 1
            self.feature_counts[label] += features
        
        logger.info(f"Training complete. Classes: {self.classes}, Vocab size: {self.vocab_size}")
    
    def _get_class_probability(self, label: str) -> float:
        """Get P(class)"""
        return self.class_counts[label] / self.total_docs
    
    def _get_feature_probability(self, feature_idx: int, label: str) -> float:
        """Get P(feature|class) with Laplace smoothing"""
        
        count = self.feature_counts[label][feature_idx]
        total_count = np.sum(self.feature_counts[label])
        
        # Laplace smoothing
        numerator = count + self.alpha
        denominator = total_count + self.alpha * self.vocab_size
        
        return numerator / denominator
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get class probabilities
        
        Args:
            X: Feature matrix
        
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        X = np.array(X)
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, len(self.classes)))
        
        for i, features in enumerate(X):
            for j, label in enumerate(self.classes):
                class_prob = self._get_class_probability(label)
                
                if self.log_space:
                    # Log-space to avoid underflow
                    log_prob = np.log(class_prob)
                    
                    for feat_idx in range(len(features)):
                        if features[feat_idx] > 0:
                            feat_prob = self._get_feature_probability(feat_idx, label)
                            log_prob += features[feat_idx] * np.log(feat_prob)
                    
                    probas[i, j] = np.exp(log_prob)
                else:
                    # Regular space
                    prob = class_prob
                    
                    for feat_idx in range(len(features)):
                        if features[feat_idx] > 0:
                            feat_prob = self._get_feature_probability(feat_idx, label)
                            prob *= (feat_prob ** features[feat_idx])
                    
                    probas[i, j] = prob
        
        # Normalize probabilities
        probas = probas / np.sum(probas, axis=1, keepdims=True)
        
        return probas
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        
        probas = self.predict_proba(X)
        predictions = self.classes[np.argmax(probas, axis=1)]
        
        return predictions
    
    def alpha_sensitivity_analysis(self, X_test: np.ndarray, y_test: np.ndarray,
                                  alphas: List[float] = None) -> Dict:
        """
        Analyze sensitivity to alpha (smoothing) parameter
        
        Args:
            X_test: Test features
            y_test: Test labels
            alphas: List of alpha values to test
        
        Returns:
            Sensitivity analysis results
        """
        
        if alphas is None:
            alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
        
        logger.info(f"Performing alpha sensitivity analysis: {alphas}")
        
        results = {"alpha": [], "accuracy": [], "f1_score": []}
        
        for alpha in alphas:
            # Create model with this alpha
            model = MultinomialNaiveBayes(alpha=alpha)
            model.class_counts = self.class_counts.copy()
            model.feature_counts = {k: v.copy() for k, v in self.feature_counts.items()}
            model.vocab_size = self.vocab_size
            model.classes = self.classes
            model.total_docs = self.total_docs
            model.feature_names = self.feature_names
            
            # Predict
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
            
            results["alpha"].append(alpha)
            results["accuracy"].append(acc)
            results["f1_score"].append(f1)
        
        logger.info(f"Best alpha: {alphas[np.argmax(results['f1_score'])]}")
        
        return results


# ========== TASK 5.2: LOGISTIC REGRESSION ==========

class LogisticRegressionClassifier:
    """
    Logistic Regression with L1, L2, and ElasticNet regularization
    
    Why LR handles correlated features better than Naive Bayes:
    1. Naive Bayes assumes feature independence (naive assumption)
    2. LR directly learns feature weights and correlations
    3. Regularization (L1/L2) explicitly handles feature correlation
    4. LR is discriminative, learns P(class|features) directly
    """
    
    def __init__(self, regularization: str = 'l2', C: float = 1.0, class_weight=None):
        """
        Initialize Logistic Regression
        
        Args:
            regularization: 'l1', 'l2', or 'elasticnet'
            C: Inverse regularization strength
        """
        
        self.regularization = regularization
        self.C = C
        
        self.models = {}  # One model per class
        self.classes = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.X_train_scaled = None
        self.class_weight = class_weight
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """Train Logistic Regression models"""
        
        logger.info(f"Training Logistic Regression ({self.regularization} regularization)...")
        
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.X_train_scaled = X_scaled
        
        self.classes = np.unique(y)
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Determine penalty type
        penalty = 'l1' if self.regularization == 'l1' else 'l2'
        solver = 'saga' if self.regularization == 'l1' else 'lbfgs'
        
        # Handle ElasticNet
        if self.regularization == 'elasticnet':
            penalty = 'elasticnet'
            solver = 'saga'
        
        # Train one-vs-rest classifier
        for cls in self.classes:
            y_binary = (y == cls).astype(int)
            
            model = SklearnLR(
                C=self.C,
                penalty=penalty,
                solver=solver,
                max_iter=1000,
                random_state=42,
                l1_ratio=0.5 if self.regularization == 'elasticnet' else None,
                class_weight=self.class_weight,
            )
            model.fit(X_scaled, y_binary)
            
            self.models[cls] = model
        
        logger.info(f"Training complete. Classes: {self.classes}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities for each class
        probas = np.zeros((X.shape[0], len(self.classes)))
        
        for i, cls in enumerate(self.classes):
            probas[:, i] = self.models[cls].predict_proba(X_scaled)[:, 1]
        
        # Predict class with max probability
        predictions = self.classes[np.argmax(probas, axis=1)]
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get probability predictions"""
        
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        X_scaled = self.scaler.transform(X)
        
        probas = np.zeros((X.shape[0], len(self.classes)))
        
        for i, cls in enumerate(self.classes):
            probas[:, i] = self.models[cls].predict_proba(X_scaled)[:, 1]
        
        # Normalize
        probas = probas / np.sum(probas, axis=1, keepdims=True)
        
        return probas
    
    def get_top_features(self, class_label: str, top_n: int = 20) -> List[Tuple]:
        """Get top weighted features for a class"""
        
        model = self.models[class_label]
        coefficients = model.coef_[0]
        
        top_indices = np.argsort(np.abs(coefficients))[-top_n:][::-1]
        
        results = []
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            weight = coefficients[idx]
            results.append((feature_name, weight))
        
        return results

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        return {
            "accuracy": float(accuracy_score(y, predictions)),
            "precision_weighted": float(precision_score(y, predictions, average='weighted', zero_division=0)),
            "recall_weighted": float(recall_score(y, predictions, average='weighted', zero_division=0)),
            "f1_weighted": float(f1_score(y, predictions, average='weighted', zero_division=0)),
            "confusion_matrix": confusion_matrix(y, predictions).tolist(),
            "classification_report": classification_report(y, predictions, output_dict=True, zero_division=0),
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
        }

    def compute_roc_data(self, X: np.ndarray, y: np.ndarray) -> Dict:
        probabilities = self.predict_proba(X)
        roc_data = {}
        for idx, cls in enumerate(self.classes):
            binary_truth = (np.array(y) == cls).astype(int)
            fpr, tpr, _ = roc_curve(binary_truth, probabilities[:, idx])
            roc_data[str(cls)] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(auc(fpr, tpr)),
            }
        return roc_data


# ========== TASK 5.3: POLYNOMIAL FEATURES + LR ==========

class PolynomialLRClassifier:
    """
    Polynomial Features with Logistic Regression
    
    Process:
    1. Reduce TF-IDF to 2D with PCA
    2. Apply polynomial features (degree 1, 2, 3)
    3. Train Logistic Regression
    4. Plot decision boundaries
    """
    
    def __init__(self, n_components: int = 2):
        """Initialize"""
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.models_by_degree = {}
        self.scalers_by_degree = {}
        self.feature_names = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train models with different polynomial degrees"""
        
        logger.info("Training Polynomial LR models...")
        
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Reduce to 2D
        X_2d = self.pca.fit_transform(X)
        
        logger.info(f"PCA explained variance: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        
        # Train models with different degrees
        for degree in [1, 2, 3]:
            logger.info(f"Training degree-{degree} polynomial model...")
            
            # Create polynomial features
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_poly = poly.fit_transform(X_2d)
            
            # Scale
            scaler = StandardScaler()
            X_poly_scaled = scaler.fit_transform(X_poly)
            
            # Train LR
            model = SklearnLR(max_iter=1000, random_state=42)
            model.fit(X_poly_scaled, y)
            
            self.models_by_degree[degree] = {
                "model": model,
                "poly": poly,
                "scaler": scaler,
                "X_2d": X_2d,
                "y_train": np.array(y),
                "X_poly_scaled": X_poly_scaled,
            }
            
            # Compute feature space size
            feature_space_size = X_poly.shape[1]
            logger.info(f"Degree {degree} feature space size: {feature_space_size}")
    
    def predict(self, X: np.ndarray, degree: int = 2) -> np.ndarray:
        """Predict with specified degree"""
        
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        X_2d = self.pca.transform(X)
        
        model_info = self.models_by_degree[degree]
        X_poly = model_info["poly"].transform(X_2d)
        X_poly_scaled = model_info["scaler"].transform(X_poly)
        
        predictions = model_info["model"].predict(X_poly_scaled)
        
        return predictions
    
    def evaluate_degrees(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate models across degrees"""
        
        logger.info("Evaluating polynomial degrees...")
        
        results = {}
        
        for degree in [1, 2, 3]:
            predictions = self.predict(X_test, degree=degree)
            
            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
            
            train_predictions = self.models_by_degree[degree]["model"].predict(
                self.models_by_degree[degree]["X_poly_scaled"]
            )
            results[degree] = {
                "train_accuracy": float(accuracy_score(self.models_by_degree[degree]["y_train"], train_predictions)),
                "test_accuracy": float(acc),
                "f1_score": float(f1),
                "feature_space_size": int(self.models_by_degree[degree]["poly"].n_output_features_),
            }
            
            logger.info(f"Degree {degree}: Accuracy={acc:.4f}, F1={f1:.4f}")
        
        return results
    
    def plot_decision_boundaries(self, degree: int = 2, y: np.ndarray = None,
                                save_path: str = "./decision_boundary.png"):
        """Plot decision boundaries for 2D PCA-reduced data"""
        
        logger.info(f"Plotting decision boundary for degree {degree}...")
        
        model_info = self.models_by_degree[degree]
        X_2d = model_info["X_2d"]
        
        # Create mesh
        h = 0.02
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Predict on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        X_mesh_poly = model_info["poly"].transform(mesh_points)
        X_mesh_scaled = model_info["scaler"].transform(X_mesh_poly)
        Z = model_info["model"].predict(X_mesh_scaled)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        
        if y is not None:
            scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y,
                                cmap='RdYlBu', edgecolors='black', s=50)
            plt.colorbar(scatter)
        
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'Decision Boundary (Polynomial Degree {degree})')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved to {save_path}")
        plt.close()


def main():
    """Test ML models"""
    
    # Create synthetic data
    from sklearn.datasets import make_classification
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Create text data
    texts = [
        "stock market rises government announces",
        "breaking viral news spreading false claim",
        "official statement university admissions open",
        "unconfirmed report celebrity death hoax"
    ] * 100
    
    labels = ["Real", "Fake", "Real", "Fake"] * 100
    y = np.array([0 if l == "Real" else 1 for l in labels])
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(texts).toarray()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Dataset: train={len(X_train)}, test={len(X_test)}")
    
    # Test Naive Bayes
    nb = MultinomialNaiveBayes(alpha=1.0)
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred)
    logger.info(f"Naive Bayes Accuracy: {nb_acc:.4f}")
    
    # Test Logistic Regression
    for reg in ['l2', 'l1', 'elasticnet']:
        lr = LogisticRegressionClassifier(regularization=reg)
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_acc = accuracy_score(y_test, lr_pred)
        logger.info(f"LR ({reg}) Accuracy: {lr_acc:.4f}")
    
    # Test Polynomial
    poly_lr = PolynomialLRClassifier()
    poly_lr.fit(X_train, y_train)
    poly_results = poly_lr.evaluate_degrees(X_test, y_test)
    logger.info(f"Polynomial results: {poly_results}")


if __name__ == "__main__":
    main()
