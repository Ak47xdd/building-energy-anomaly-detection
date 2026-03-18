# Building Energy Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-orange)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-yellow)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Detect anomalies in building energy consumption using ML models (Isolation Forest, LOF, Elliptic Envelope ensemble). Includes Jupyter notebooks for analysis and **FastAPI** for production API deployment.

## Table of Contents

- [About](#about)
- [Repository Structure](#structure)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Jupyter Notebooks](#notebooks)
  - [FastAPI](#api)
- [Workflow](#workflow)
- [Outputs](#outputs)
- [API Endpoints](#endpoints)

## About

Analyzes time-series energy data (electricity, water, gas, etc.) for anomalies using ensemble ML: **Isolation Forest**, **Local Outlier Factor**, **Elliptic Envelope**. Produces visualizations, results CSV, and **REST API**.

## Repository Structure

```
building-energy-anomaly-detection/
├── data/meters/raw/          # Raw CSV (electricity.csv, etc.)
├── data/meters/whole/        # Processed EDA CSV
├── notebooks/                # Analysis pipeline (01_eda.ipynb → 04_model_training.ipynb)
├── API/                      # FastAPI production API (main.py)
├── Plots/                    # Generated visualizations
├── results/                  # Anomaly detection outputs (CSV)
├── requirements.txt
└── README.md
```

## Dataset

**Raw**: `data/meters/raw/*.csv` (electricity, chilledwater, gas, etc.) - timestamp + consumption.

**Processed**: `data/meters/whole/eda.csv` - EDA-ready for modeling.

## Requirements

See `requirements.txt`:

```
fastapi
uvicorn
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
jupyter
```

## Installation

```bash
git clone <repo>
cd building-energy-anomaly-detection
python -m venv env
# Activate env...
pip install -r requirements.txt
```

## Usage

### Jupyter Notebooks

```bash
jupyter notebook
```

Run sequentially:

1. `01_eda.ipynb` - Explore distributions, correlations.
2. `02_preprocessing.ipynb` - Clean data.
3. `03_feature_engineering.ipynb` - Create features.
4. `04_model_training.ipynb` - **Train ensemble** → Save `results/anomaly_detection_results.csv` + plots.

### FastAPI

```bash
cd API
uvicorn main:app --reload
```

- http://localhost:8000/docs
- **POST /detect-anomalies** → Instant JSON results (anomalies %, top 10, votes).

## Outputs

- **Plots/**: anomaly_scores_distribution.png, feature_importance.png, anomaly_detection_plot.png.
- **results/**: anomaly_detection_results.csv (w/ `is_anomaly`, `anomaly_votes`).
- **API**: JSON summary.

## API Endpoints

| Method | Endpoint            | Description                           |
| ------ | ------------------- | ------------------------------------- |
| GET    | `/`                 | API info                              |
| POST   | `/detect-anomalies` | Run ensemble detection → JSON summary |

**Example Response**:

```json
{
  "total_points": 10000,
  "anomaly_count": 250,
  "anomaly_percentage": 2.5,
  "votes_distribution": { "0": 9500, "1": 200, "2": 150, "3": 150 },
  "top_anomalies": [
    /* first 10 anomaly rows */
  ]
}
```

**Features**: Parallel training, majority vote ensemble, Swagger UI (/docs), production-ready.
