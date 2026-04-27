# PakSentinel

PakSentinel is an end-to-end NLP pipeline for misinformation detection, semantic retrieval, MLflow tracking, and FastAPI deployment. This repository is organized around the assignment tasks and is meant to be run reproducibly from a clean virtual environment.

## What this repo now expects

This codebase no longer invents data with dummy or synthetic fallbacks. You must place real source files in `data/raw/` before running the full pipeline.

Expected source layout:

```text
data/
  raw/
    liar/
      train.tsv
      valid.tsv
      test.tsv
    isot/
      Fake.csv
      True.csv
    covid/
      covid_19_data.csv
      # or another supported COVID misinformation file
    pakistan/
      pakistan_dataset.csv
      # or pakistan_dataset.parquet / pakistan_dataset.jsonl
```

## Fresh setup

From the repository root:

```powershell
Remove-Item -Recurse -Force -LiteralPath .venv -ErrorAction SilentlyContinue
C:\Users\Muhammad Zunair\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Optional downloads for smoother first run:

```powershell
.\.venv\Scripts\python.exe -m nltk.downloader punkt stopwords wordnet
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

## Run all tasks

Run the full assignment pipeline:

```powershell
.\.venv\Scripts\python.exe main.py --output-dir .\output
```

This executes:

1. Task 1: source-backed dataset loading, deduplication, reliability report
2. Task 2: data lake uploads for raw and processed layers
3. Task 3: cleaning audit, tokenizer comparison, stopword study, normalization study, TF-IDF/Word2Vec outputs
4. Task 4: n-gram language models and held-out evaluation
5. Task 5: 3-class Naive Bayes, Logistic Regression variants, Polynomial LR, saved production bundle
6. Task 6: MLflow preprocessing ablation study and model registration flow
7. Task 7: API artifact preparation

Generated outputs go to:

```text
output/
models/
data/lake/
logs/
mlflow_data/
mlflow_artifacts/
```

## Run individual pieces

Skip selected tasks:

```powershell
.\.venv\Scripts\python.exe main.py --skip-task task6 --skip-task task7
```

Run just the API tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_api.py -q
```

## Start the API

Make sure Task 5 has already created `models/production_bundle.pkl`, then start FastAPI:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Available endpoints:

- `GET /health`
- `POST /preprocess`
- `POST /classify`
- `POST /classify/batch`
- `POST /retrieve/similar`
- `GET /model/performance`

Swagger docs:

```text
http://127.0.0.1:8000/docs
```

## Run MLflow locally

You can use either Docker or a local SQLite-backed MLflow store.

Local example:

```powershell
.\.venv\Scripts\python.exe -m mlflow server --backend-store-uri sqlite:///mlflow_data/mlflow.db --default-artifact-root .\mlflow_artifacts --host 0.0.0.0 --port 5000
```

Then browse:

```text
http://127.0.0.1:5000
```

## Docker Compose

Bring up the bundled services:

```powershell
docker-compose up -d
```

This starts PostgreSQL, MongoDB, MinIO, MLflow, FastAPI, and Jupyter.

## Notes

- `requirements.txt` was adjusted for Python 3.12 compatibility.
- The pipeline is now strict about missing data sources instead of silently fabricating them.
- API inference is model-backed and loads the saved production bundle at startup.
- The API test suite currently passes in the rebuilt virtual environment.

## Verified command

The following completed successfully after rebuilding the environment:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_api.py -q
```
