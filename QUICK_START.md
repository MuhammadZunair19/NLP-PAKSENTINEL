# PakSentinel Quick Start Guide

## 5-Minute Setup

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional)
- Git

### Option A: Docker (Recommended)

```bash
# Clone repository (if applicable)
cd /path/to/NLP/Assignment\ 2

# Start all services
docker-compose up -d

# Wait for services
sleep 30

# Run pipeline
python main.py

# Access dashboard
# - FastAPI docs: http://localhost:8000/docs
# - MLFlow UI: http://localhost:5000
# - Jupyter: http://localhost:8888
```

### Option B: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download language models
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
python -m spacy download en_core_web_sm

# Run pipeline
python main.py --output-dir ./output

# Start API
python -m uvicorn api.app:app --reload

# Run tests
pytest tests/test_api.py -v
```

## Common Commands

```bash
# Run specific task
python main.py --skip-task task1 --skip-task task2

# Run tests with coverage
pytest tests/test_api.py --cov=api --cov-report=html

# Start API with custom port
python -m uvicorn api.app:app --port 8001

# View MLFlow experiments
mlflow ui --backend-store-uri ./mlruns

# Check API health
curl http://localhost:8000/health | python -m json.tool

# Classify text
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Sample text for classification"}'
```

## File Structure

```
Assignment 2/
├── src/                    # Core modules
│   ├── data_sourcing.py            # Task 1
│   ├── data_lake_manager.py        # Task 2
│   ├── nlp_pipeline.py             # Task 3
│   ├── ngram_language_models.py    # Task 4
│   ├── ml_models.py                # Task 5
│   └── mlflow_integration.py       # Task 6
├── api/
│   └── app.py                      # Task 7 - FastAPI
├── tests/
│   └── test_api.py                 # Comprehensive tests
├── main.py                 # Pipeline orchestrator
├── README.md               # Full documentation
├── requirements.txt        # Dependencies
├── docker-compose.yml      # Container setup
└── Dockerfile              # API container
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   PakSentinel Pipeline                   │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐       │
│  │  Task 1  │  →   │  Task 2  │  →   │  Task 3  │       │
│  │  Sources │      │ Storage  │      │   NLP    │       │
│  └──────────┘      └──────────┘      └──────────┘       │
│                                           ↓               │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐       │
│  │  Task 7  │  ←   │  Task 6  │  ←   │  Task 5  │       │
│  │   API    │      │ MLFlow   │      │   ML     │       │
│  └──────────┘      └──────────┘      └──────────┘       │
│       ↑                  ↑                  ↑             │
│       └──────────────────┼──────────────────┘             │
│              Task 4: N-gram Models                        │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Validation Checklist

After setup, verify:

```bash
# 1. Data Sourcing
ls -la ./data/raw/
ls -la ./output/task1_*.json

# 2. Storage
ls -la ./data/lake/

# 3. NLP Pipeline
ls -la ./output/task3_*.json

# 4. ML Models
ls -la ./output/task5_*.json

# 5. API Health
curl http://localhost:8000/health

# 6. Tests Pass
pytest tests/test_api.py -v

# 7. Summary Report
cat ./output/summary_report.json
```

## Troubleshooting

### API Won't Start
```bash
# Check if port is in use
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Use different port
python -m uvicorn api.app:app --port 8001
```

### Missing Data
```bash
# Create output directories
mkdir -p data/raw data/processed data/lake data/embeddings output
```

### Docker Issues
```bash
# Force rebuild
docker-compose down -v
docker-compose up -d --build

# Check logs
docker-compose logs -f mlflow
```

### NLTK/Spacy Data
```bash
python -c "
import nltk
import spacy

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

try:
    spacy.load('en_core_web_sm')
except:
    print('Run: python -m spacy download en_core_web_sm')
"
```

## Next Steps

1. **Review Results**: Check `output/summary_report.json`
2. **Explore Data**: See `data/raw/combined_dataset.parquet`
3. **View Experiments**: Open MLFlow UI at http://localhost:5000
4. **Test API**: Visit http://localhost:8000/docs
5. **Run Notebooks**: Start Jupyter Lab at http://localhost:8888

## Performance Metrics

Expected performance on standard dataset:

| Task | Component | Metric | Target |
|------|-----------|--------|--------|
| 1 | Data Sourcing | Dataset Size | 5,000+ |
| 1 | Reliability | Avg Score | 3.0+ /5 |
| 3 | Tokenization | Speed | <1ms/doc |
| 5 | NB Classifier | Accuracy | 80%+ |
| 5 | LR Classifier | Accuracy | 85%+ |
| 7 | /classify | Response Time | <100ms |
| 7 | /batch (10 texts) | Response Time | <200ms |

## Support

- Check inline documentation in source files
- Review comprehensive README.md
- Run test suite: `pytest tests/ -v`
- Check logs: `./logs/` directory

---

**Happy hacking! 🚀**
