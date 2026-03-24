# Building Energy Anomaly Detection & RAG Assistant

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-orange)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-yellow)](https://scikit-learn.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-purple)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Dual-purpose repo**:

1. **ML Anomaly Detection** in building energy consumption (Isolation Forest + LOF + Elliptic Envelope ensemble).
2. **RAG Assistant** for querying energy audit PDF reports using LangChain + Groq + Chroma.

Includes Jupyter notebooks, **FastAPI APIs** (anomaly detection + RAG chat), and production deployment.

## Table of Contents

- [About](#about)
- [Repository Structure](#structure)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Jupyter Notebooks (Anomaly Detection)](#notebooks)
  - [FastAPI Anomaly Detection](#api-anomaly)
  - [RAG Assistant](#rag)
  - [FastAPI RAG API](#api-rag)
- [Workflow](#workflow)
- [Outputs](#outputs)
- [API Endpoints](#endpoints)

## About

**Anomaly Detection**: Analyzes time-series energy data (electricity, water, gas, etc.) for anomalies using ML ensemble. Produces visualizations, CSV results, REST API.

**RAG Assistant**: Loads energy audit PDFs → creates vector store → answers questions via Groq LLM + LangChain retrieval.

## Repository Structure

```
building-energy-anomaly-detection/
├── data/meters/raw/              # Raw CSV (electricity.csv, etc.)
├── data/meters/whole/            # Processed EDA CSV
├── documents/                    # Energy audit PDFs
├── notebooks/                    # ML analysis (01_eda.ipynb → 04_model_training.ipynb)
├── API/                          # FastAPI apps
│   ├── main.py                   # Anomaly detection API
│   └── app.py                    # RAG API
├── RAG/                          # RAG pipeline
│   └── rag.py                    # LangChain RAG implementation
├── Plots/                        # ML visualizations
├── results/                      # Anomaly results CSV
├── chroma_db/                    # Vector store (auto-created)
├── requirements.txt              # ML deps
├── rag_requirements.txt          # RAG deps
├── README.md
└── .env.example                  # GROQ_API_KEY=your_key_here
```

## Dataset

**Energy Meters**: `data/meters/raw/*.csv` (electricity, chilledwater, gas, hotwater, irrigation, solar, steam, water).

**Processed**: `data/meters/whole/eda.csv`.

**Documents**: `documents/*.pdf` - Energy audit reports for RAG.

## Requirements

**ML Anomaly Detection**: `pip install -r requirements.txt`

**RAG**: `pip install -r rag_requirements.txt`

Key deps:

```
# ML
fastapi uvicorn pandas scikit-learn matplotlib seaborn

# RAG
langchain-community langchain-groq chromadb sentence-transformers pypdf
```

**Groq API Key**: Required for RAG. Get free at https://console.groq.com/keys

```
# .env
GROQ_API_KEY=your_key_here
```

## Installation

```bash
git clone https://github.com/Ak47xdd/building-energy-anomaly-detection
cd building-energy-anomaly-detection

# ML env
python -m venv rag_local_env
# Activate...
pip install -r requirements.txt

# RAG env (recommended)
python -m venv rag_local_env
# Activate...
pip install -r rag_requirements.txt
```

## Usage

### Jupyter Notebooks (Anomaly Detection)

```bash
jupyter notebook
```

1. `01_eda.ipynb` - EDA.
2. `02_preprocessing.ipynb` - Cleaning.
3. `03_feature_engineering.ipynb` - Features.
4. `04_model_training.ipynb` - Train ensemble → `results/anomaly_detection_results.csv` + plots.

### FastAPI Anomaly Detection

```bash
cd API
uvicorn main:app --reload
```

Visit: http://localhost:8000/docs

**POST /detect-anomalies** → JSON anomaly summary.

### RAG Assistant (CLI)

```bash
cd RAG
python rag.py
```

Interactive chat over PDFs. Type questions like:

- "What are the main energy saving recommendations?"
- Sources cited automatically.

```
{
  "answer": "The audit recommends upgrading HVAC systems in Building A
             and installing smart meters across all floors to reduce
             consumption by an estimated 18%.",
  "sources": ["documents/building_audit_2024.pdf"]
}
```

**First run**: Builds `chroma_db/` (~1-2 min).

### FastAPI RAG API

```bash
cd API
uvicorn app:app --reload --env-file ../.env
```

- http://localhost:8000/docs
- **POST /query** `{ "question": "..." }` → JSON answer + sources.

## Workflow

1. **ML**: notebooks → train → API/main.py → results/plots.
2. **RAG**: PDFs → rag.py (builds chroma_db) → API/app.py → chat/query.

**Production**: Both APIs can run concurrently (different ports).

## Outputs

**ML**:

- `Plots/*.png` (anomaly plots, heatmaps).
- `results/anomaly_detection_results.csv` (`is_anomaly`, `anomaly_votes`).

**RAG**:

- `chroma_db/` (persistent vector store).
- JSON responses w/ answers + source PDFs.

## API Endpoints

### Anomaly Detection (main.py)

| Method | Endpoint            | Description               |
| ------ | ------------------- | ------------------------- |
| GET    | `/`                 | Info                      |
| POST   | `/detect-anomalies` | Ensemble detection → JSON |

### RAG Assistant (app.py)

| Method | Endpoint | Description                          |
| ------ | -------- | ------------------------------------ |
| GET    | `/`      | "Energy Audit RAG Assistant running" |
| POST   | `/query` | `{question: str}` → Answer + sources |

**RAG Example**:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What energy efficiency measures are recommended?"}'
```

**Features**: ML ensemble voting, PDF RAG w/ local embeddings, dual FastAPI apps, Swagger docs, env isolation, free Groq inference.
