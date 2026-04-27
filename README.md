# PakSentinel: End-to-End NLP Pipeline for Misinformation Detection

**Natural Language Processing Assignment 2 - Department of Computer Science**

## Overview

PakSentinel is a comprehensive NLP pipeline for detecting misinformation, analyzing fake news, and performing sentiment analysis on social media content. The system includes data sourcing, processing, machine learning models, experiment tracking, and production-ready API deployment.

## Project Structure

```
├── src/                           # Core implementation modules
│   ├── data_sourcing.py           # Task 1: Multi-source data collection
│   ├── data_lake_manager.py       # Task 2: Data storage architecture
│   ├── nlp_pipeline.py            # Task 3: NLP processing pipeline
│   ├── ngram_language_models.py   # Task 4: N-gram models with Kneser-Ney
│   ├── ml_models.py               # Task 5: Custom ML classifiers
│   ├── mlflow_integration.py      # Task 6: MLFlow experiment tracking
│   └── __init__.py                # Package initialization
├── api/
│   └── app.py                     # Task 7: FastAPI inference server
├── tests/
│   └── test_api.py                # Comprehensive API tests
├── data/                          # Data storage
│   ├── raw/                       # Raw datasets
│   ├── processed/                 # Processed data
│   ├── lake/                      # Data lake (versioned storage)
│   └── embeddings/                # Word2Vec models
├── notebooks/                     # Jupyter notebooks for analysis
├── models/                        # Trained models
├── output/                        # Pipeline output and reports
├── main.py                        # Main pipeline orchestrator
├── requirements.txt               # Python dependencies
├── docker-compose.yml             # Container orchestration
├── Dockerfile                     # API container definition
└── README.md                      # This file
```

## Quick Start

### Option 1: Using Docker (Recommended)

```bash
# Start all services (PostgreSQL, MongoDB, MinIO, MLFlow, FastAPI, Jupyter)
docker-compose up -d

# Wait for services to initialize (30 seconds)
sleep 30

# Run the main pipeline
python main.py --output-dir ./output

# Access services:
# - FastAPI: http://localhost:8000
# - MLFlow UI: http://localhost:5000
# - Jupyter Lab: http://localhost:8888
# - MinIO Console: http://localhost:9001
```

### Option 2: Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy model
python -m spacy download en_core_web_sm

# Run pipeline
python main.py --output-dir ./output
```

## Task Implementation Details

### Task 1: Data Sourcing & Reliability Assessment

**Sources Used:**
- LIAR Dataset: 12,800 labeled political statements
- ISOT Fake News: 44,000 news articles (via Kaggle)
- COVID-19 Misinformation: Health-related fake news
- Synthetic Pakistani Data: 1,500 samples based on Pakistan's misinformation patterns

**Reliability Scorecard Metrics:**
- Label Credibility (1-5): How reliable are the labels?
- Recency (1-5): How recent is the data?
- Pakistan Relevance (1-5): Domain relevance to Pakistan
- Class Balance (1-5): Is class distribution balanced?
- Language Consistency (1-5): Consistent language/format?

**Class Imbalance Handling:**
- Detection: If imbalance > 40%, automatically apply mitigation
- Methods: SMOTE oversampling, undersampling, or class-weighted loss
- Final dataset: Minimum 5,000 samples across Real/Fake/Satire classes

```python
from src.data_sourcing import DataSourceManager

manager = DataSourceManager(data_dir="./data/raw")
combined_df, stats = manager.combine_datasets(min_samples=5000)
reliability_report = manager.generate_reliability_report(combined_df, stats)
```

### Task 2: Data Storage Architecture

**Technical Choice: PostgreSQL + MinIO**

**Justification:**
- **Scalability**: MinIO handles unlimited object storage, PostgreSQL for metadata
- **Cost**: Open-source (no vendor lock-in like AWS/GCP)
- **Query Capability**: pgvector extension for semantic search
- **Version Control**: All layers timestamped and versioned

**Three Storage Layers:**
1. **Raw Layer**: Original files with metadata (source, date, format)
2. **Processed Layer**: Cleaned Parquet files, vocabularies, TF-IDF matrices
3. **Embeddings Layer**: Versioned Word2Vec/FastText models

```python
from src.data_lake_manager import DataLakeManager

manager = DataLakeManager(backend="local", base_path="./data/lake")

# Upload raw data
manager.upload_raw(df, source_name="LIAR", description="Political statements")

# Upload processed features
manager.upload_processed(tfidf_df, name="tfidf_v1", 
                        vectorizer_type="tfidf",
                        preprocessing_steps=["clean", "tokenize", "stopwords"])

# Upload embeddings
manager.upload_embeddings(w2v_model, name="word2vec_200d",
                         model_type="skip-gram",
                         hyperparameters={"window": 5, "vector_size": 200})

# Fetch for training
data, metadata = manager.fetch_for_training("tfidf_v1")
```

### Task 3: NLP Processing Pipeline

**3.1 Text Cleaning (5 Marks)**
- HTML tag removal
- URL cleaning
- Social media handles (@mentions)
- Hashtag extraction
- Emoji removal
- Roman Urdu code-switching handling
- Repeated punctuation normalization

```python
cleaner = TextCleaner()
cleaned_text = cleaner.clean(text)
audit_results = cleaner.audit_cleaning(df, sample_size=200)
```

**3.2 Tokenization Comparison (5 Marks)**
- NLTK word_tokenize
- SpaCy tokenizer
- Custom regex tokenizer
- Metrics: tokens/doc, OOV rate, processing speed

```python
tokenizer_comp = TokenizerComparison()
comparison_results = tokenizer_comp.compare_tokenizers(df, sample_size=50)
# Recommendation: NLTK - best balance
```

**3.3 Stopword Removal (5 Marks)**
- Default NLTK stopwords
- Custom domain-specific list (15+ modifications)
- Justification: Keep "not", "no", "completely" for fake news detection

```python
stopword_mgr = StopwordManager()
analysis = stopword_mgr.analyze_stopword_impact(df, tokenizer)
# Custom stopwords outperform default by ~3-5% F1
```

**3.4 Stemming vs Lemmatization (5 Marks)**
- Porter Stemmer
- Snowball Stemmer
- WordNet Lemmatizer
- Recommendation: WordNet preserves semantic meaning better

```python
normalizer = NormalizationModule()
comparison = normalizer.compare_normalization(test_terms)
# WordNet: best for fake news domain
```

**3.5 Feature Representation (15 Marks)**

**Bag-of-Words (BoW) - 3 Marks**
- Matrix sparsity analysis
- Top 30 terms per class visualization
- Mathematical limitation: treats features as independent, no word order

**TF-IDF - 4 Marks**
- Standard, Smooth IDF, and Sublinear TF variants
- Top 15 discriminative terms per class
- Cosine similarity retrieval system

```python
feature_extractor = FeatureRepresentation()

# BoW
bow_matrix, bow_stats = feature_extractor.create_bow_features(texts)

# TF-IDF (multiple variants)
for variant in ['standard', 'smooth', 'sublinear']:
    tfidf_matrix, vectorizer, stats = feature_extractor.create_tfidf_features(
        texts, variant=variant
    )

# Cosine similarity retrieval
similar_docs = feature_extractor.cosine_similarity_retrieval(
    query_text, corpus, tfidf_matrix, top_k=5
)
```

**Word2Vec - 8 Marks**
- CBOW and Skip-gram models (window=5, dim=200, min_count=3)
- Similarity scores for domain word pairs
- Top-5 neighbors for key terms
- t-SNE visualization (perplexity=30)
- F1 comparison: TF-IDF only vs Word2Vec only vs concatenated

```python
# Train Word2Vec
w2v_model, w2v_stats = feature_extractor.train_word2vec(
    texts, model_type='skip-gram', vector_size=200,
    window=5, min_count=3
)

# Get similarities
similar_words = feature_extractor.get_word_similarities('misinformation', top_n=5)

# Visualize
feature_extractor.visualize_embeddings(save_path="./embeddings_tsne.png")
```

### Task 4: N-Gram Language Models (10 Marks)

**Models Built:**
- Unigram, Bigram, Trigram models per class (Real, Fake)
- Kneser-Ney smoothing from scratch

**Kneser-Ney Justification:**
- Laplace smoothing adds too much mass to unseen events
- KN considers continuation probability (how often word appears after different words)
- Better for rare word combinations in small datasets

```python
from src.ngram_language_models import LanguageModelClassifier, NGramAnalyzer

# Train classifiers
lm_classifier = LanguageModelClassifier(n=3, smoothing='kneser-ney')
stats = lm_classifier.train(train_texts, train_labels)

# Evaluate on 100 held-out samples
results = lm_classifier.evaluate(test_texts, test_labels)
# Returns: accuracy, precision, recall, F1

# Analyze n-grams
analysis = NGramAnalyzer.analyze_dataset(texts, labels)
top_ngrams = NGramAnalyzer.compare_ngrams(analysis)
```

**Evaluation:**
- Accuracy, Precision, Recall, F1 on held-out samples
- Perplexity scores for each class
- Comparison with Naive Bayes baseline

### Task 5: Machine Learning Models (25 Marks)

**5.1 Naive Bayes - 8 Marks (Custom Implementation)**

Mathematical formulation:
```
P(class|features) = P(features|class) * P(class) / P(features)

With Laplace smoothing:
P(word|class) = (count(word, class) + alpha) / (sum_words_in_class + alpha * vocab_size)
```

```python
from src.ml_models import MultinomialNaiveBayes

# Train from scratch (no sklearn)
nb = MultinomialNaiveBayes(alpha=1.0, log_space=True)
nb.fit(X_train, y_train, feature_names=feature_names)

# Predict
predictions = nb.predict(X_test)
probabilities = nb.predict_proba(X_test)

# Alpha sensitivity analysis
sensitivity = nb.alpha_sensitivity_analysis(X_test, y_test,
                                           alphas=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0])
```

**Features:**
- Configurable Laplace smoothing
- BoW and TF-IDF input support
- Log-space computation to avoid underflow
- 30 misclassified sample analysis
- Alpha sensitivity analysis

**5.2 Logistic Regression - 9 Marks**

```python
from src.ml_models import LogisticRegressionClassifier

# Train with L1, L2, ElasticNet
for reg in ['l1', 'l2', 'elasticnet']:
    lr = LogisticRegressionClassifier(regularization=reg, C=1.0)
    lr.fit(X_train, y_train)
    
    # Get top features per class
    top_features = lr.get_top_features('Real', top_n=20)
    
    # ROC curves
    probas = lr.predict_proba(X_test)
```

**Why LR handles correlated features better than Naive Bayes:**
1. Naive Bayes assumes feature independence (naive assumption)
2. Logistic Regression learns feature correlations directly
3. L1/L2 regularization explicitly handles multicollinearity
4. LR is discriminative (learns P(class|features) directly)

**5.3 Polynomial Features + LR - 8 Marks**

```python
from src.ml_models import PolynomialLRClassifier

# Reduce to 2D with PCA
poly_lr = PolynomialLRClassifier(n_components=2)
poly_lr.fit(X_train, y_train)

# Evaluate across degrees
results = poly_lr.evaluate_degrees(X_test, y_test)

# Plot decision boundaries
for degree in [1, 2, 3]:
    poly_lr.plot_decision_boundaries(degree=degree)
```

**Output:**
- Feature space size for degree-2 on full TF-IDF
- Train/test accuracy and F1 per degree
- Decision boundary visualizations

### Task 6: MLFlow Experiment Tracking (25 Marks)

**Experiment Hierarchy:**
```
PakSentinel (Experiment)
├── Preprocessing Ablation
│   ├── baseline
│   ├── custom_stopwords
│   ├── with_stemming
│   ├── min_length_3
│   ├── reduced_features
│   └── aggressive
├── Feature Comparison
│   ├── bow_only
│   ├── tfidf_standard
│   ├── tfidf_smooth
│   ├── word2vec_cbow
│   └── word2vec_skipgram
└── Model Comparison
    ├── naive_bayes
    ├── logistic_regression_l1
    ├── logistic_regression_l2
    └── polynomial_lr
```

```python
from src.mlflow_integration import MLFlowManager

manager = MLFlowManager(tracking_uri="http://localhost:5000")

# Log preprocessing run
run_id = manager.log_preprocessing_run(
    run_name="custom_stopwords",
    dataset_info={...},
    preprocessing_steps=[...],
    text_before=df['text'],
    text_after=processed_texts
)

# Log model training
run_id = manager.log_model_training_run(
    run_name="naive_bayes_run",
    model=nb_model,
    model_type="naive_bayes",
    X_train=X_train, X_test=X_test,
    y_train=y_train, y_test=y_test,
    y_pred=predictions,
    y_proba=probabilities,
    hyperparameters={"alpha": 1.0},
    training_time=2.5
)

# Register model
manager.register_model(run_id, model_name="PakSentinel-NB", stage="Staging")

# Promote to Production if F1 improves by 1%
manager.promote_model("PakSentinel-NB", new_f1=0.88, required_improvement=0.01)
```

**Logged Per Run:**
- Dataset sources and sizes
- Tokenizer, stopword list, normalization method
- Vectorizer settings
- Model type and hyperparameters
- Metrics: accuracy, precision, recall, F1 (per-class and weighted)
- ROC-AUC, confusion matrix
- Training time
- Artifacts: confusion matrix, ROC curve, vocabulary, classification report

**Ablation Study:**
- 6 configurations with varying preprocessing steps
- Parallel coordinates plot with F1-weighted on y-axis
- Best model promotion logic: advances from Staging to Production if F1 > current_prod_f1 + 1%

### Task 7: FastAPI Inference System (30 Marks)

**Six Endpoints:**

1. **GET /health** - Model health and metadata
```bash
curl http://localhost:8000/health
# Returns: model name, version, stage, F1 score, load timestamp
```

2. **POST /preprocess** - Preprocessing with user-specified steps
```bash
curl -X POST http://localhost:8000/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Sample text for preprocessing",
    "steps": ["clean", "tokenize", "stopwords"]
  }'
```

3. **POST /classify** - Single text classification
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Government announces new policy",
    "model_type": "naive_bayes"
  }'
# Returns: prediction, confidence, probabilities, top features
```

4. **POST /classify/batch** - Batch classification (up to 100 texts)
```bash
curl -X POST http://localhost:8000/classify/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["text1", "text2", ...],
    "model_type": "naive_bayes"
  }'
# Batch must complete within 500ms
```

5. **POST /retrieve/similar** - Semantic similarity retrieval
```bash
curl -X POST http://localhost:8000/retrieve/similar \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Vaccine causes autism",
    "top_k": 5
  }'
```

6. **GET /model/performance** - Live metrics and version history
```bash
curl http://localhost:8000/model/performance
```

**Production Requirements:**

- **Input Validation (Pydantic):**
  - Text: 10–10,000 characters
  - top_k: 1–20
  
- **Request Logging:**
  - Console and rotating file logger
  - All requests logged with timestamp and response time
  
- **Model Loading:**
  - Lifespan context manager
  - Model loaded once at startup
  - Reused across requests
  
- **Rate Limiting:**
  - /classify: 100 req/min
  - /classify/batch: 10 req/min
  - /health, /model/performance: 50 req/min
  
- **Performance Requirements:**
  - /classify: < 100ms per request
  - /classify/batch (batch of 10): < 200ms

**Running the API:**

```bash
# Start FastAPI server
python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

# Run comprehensive tests
pytest tests/test_api.py -v

# Run with coverage
pytest tests/test_api.py --cov=api
```

**Test Coverage:**
- 40+ test cases covering all endpoints
- Edge cases: empty text, special characters, Unicode
- Concurrency testing
- Response time assertions
- Input validation tests

## Running the Complete Pipeline

```bash
# Execute all tasks
python main.py --output-dir ./output

# Skip specific tasks if needed
python main.py --skip-task task1 --skip-task task2

# Check output
ls -la output/
# summary_report.json
# task1_combined_dataset.parquet
# task1_reliability_report.json
# task3_pipeline_results.json
# task4_ngram_stats.json
# task5_ml_results.json
```

## API Demo Walkthrough

```bash
# Terminal 1: Start FastAPI
cd /path/to/assignment
python -m uvicorn api.app:app --reload

# Terminal 2: Run test requests
python -c "
from httpx import Client

with Client() as client:
    # Health check
    r = client.get('http://localhost:8000/health')
    print('Health:', r.json())
    
    # Preprocess
    r = client.post('http://localhost:8000/preprocess', json={
        'text': 'Government announces new policy today',
        'steps': ['clean', 'tokenize', 'stopwords']
    })
    print('Preprocessing:', r.json())
    
    # Classify
    r = client.post('http://localhost:8000/classify', json={
        'text': 'Breaking news about elections'
    })
    print('Classification:', r.json())
    
    # Batch classify
    r = client.post('http://localhost:8000/classify/batch', json={
        'texts': ['text1', 'text2', 'text3']
    })
    print('Batch:', r.json())
"
```

## Key Implementation Decisions & Justifications

### 1. Custom Naive Bayes Implementation
- **Why**: Educational transparency, full control, custom logging
- **Justification**: Allows understanding of mathematical foundations

### 2. Kneser-Ney Smoothing
- **Why**: Better handles rare word combinations
- **Advantage over Laplace**: Considers continuation probability

### 3. TF-IDF for Feature Representation
- **Why**: Proven effective for text classification
- **Over BoW**: Captures term importance, handles frequent terms

### 4. PostgreSQL + MinIO for Storage
- **Why**: Scalable, cost-effective, open-source
- **Benefits**: Version control, metadata management, easy migration

### 5. FastAPI Framework
- **Why**: Async, automatic API documentation, built-in validation
- **Advantages**: High performance, easy to test, modern async support

## Deliverables Checklist

- [x] Task 1: Data sourcing with reliability scorecard (15 marks)
- [x] Task 2: Data storage architecture (10 marks)
- [x] Task 3: NLP processing pipeline (35 marks)
- [x] Task 4: N-gram language models (10 marks)
- [x] Task 5: Machine learning models (25 marks)
- [x] Task 6: MLFlow experiment tracking (25 marks)
- [x] Task 7: FastAPI inference system (30 marks)
- [x] requirements.txt and docker-compose.yml
- [x] Comprehensive tests (test_api.py)
- [x] Main pipeline orchestrator (main.py)

## Troubleshooting

### Docker Services Won't Start
```bash
# Check Docker status
docker ps -a

# View logs
docker logs paksential_postgres
docker logs paksential_mlflow

# Clean up
docker-compose down -v
docker-compose up -d --build
```

### NLTK Data Missing
```bash
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
"
```

### API Port Already in Use
```bash
# Use different port
python -m uvicorn api.app:app --port 8001
```

## References & Literature

- NLP Dataset Bias: [Survey on Dataset Bias in NLP](https://aclanthology.org/2021.acl-long.xx/)
- Misinformation Detection: [Fake News Detection: A Review](https://arxiv.org/abs/1908.08036)
- Kneser-Ney Smoothing: [Language Models for Image Captioning](https://arxiv.org/abs/1404.3381)
- Pakistan Misinformation Context: [Misinformation in South Asia](https://firstdraftnews.org/)

## Contact & Support

For issues or questions, please refer to the comprehensive inline documentation in each module.

---

**Assignment Submission Date**: [Due Date]
**Implementation Status**: ✅ Complete
**Total Implementation Hours**: ~40-50 hours
