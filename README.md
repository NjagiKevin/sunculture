# SunCulture — Senior Data Scientist Assessment

Customer segmentation (KMeans), credit risk analysis, and an AI-powered NL-to-SQL data platform — built for SunCulture's solar energy portfolio across Kenya, Uganda, and Côte d'Ivoire.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  STREAMLIT APP                        │
│  LangGraph NL-to-SQL (GPT-4o) → SQLite → Plotly      │
│  └─ Generate → Validate → Execute → Interpret        │
│  └─ Persistent chat history (SQLite)                  │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                  BENTOML API                          │
│  /segment_single  /segment_batch  /health  /profile   │
│  └─ Loads models/scaler.pkl + segmentation_model.pkl  │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                  AIRFLOW DAG                          │
│  Weekly retraining pipeline (champion-challenger)      │
│  └─ Extract → Train → Validate → Deploy               │
│  └─ MLflow tracking + wandb.ai logging                │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                  SOURCE CODE                          │
│  src/data/      — DataLoader, DataPreprocessor        │
│  src/features/  — FeatureEngineer                     │
│  src/models/    — CustomerSegmentation, CreditRisk    │
│  src/viz/       — Plotter (Plotly charts)             │
└──────────────────────────────────────────────────────┘
```

## Prerequisites

- Python **3.11+**
- Docker Desktop (for BentoML API & Airflow)
- An **OpenAI API key** (for the Streamlit chatbot)
- The original Excel dataset: `Senior_Data_Scientist_Assessment_Data.xlsx`

## Quick Start

### 1. Clone & setup

```bash
git clone https://github.com/NjagiKevin/sunculture.git
cd sunculture
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Place the dataset

```bash
cp /path/to/Senior_Data_Scientist_Assessment_Data.xlsx data/
```

### 3. Environment variables

Create a `.env` file (copy the template):

```bash
cp .env.example .env   # or create from scratch
```

Required variables:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | GPT-4o for the Streamlit AI chatbot |
| `WANDB_API_KEY` | Experiment logging (optional, for Airflow) |

### 4. Run the notebooks (in order)

```bash
jupyter notebook notebooks/
```

| Order | Notebook | What it does |
|-------|----------|-------------|
| 1 | `01_eda_and_storytelling.ipynb` | Exploratory analysis, 5 business insights |
| 2 | `02_customer_segmentation.ipynb` | KMeans clustering (K=4), segment profiling, marketing strategy, credit risk analysis |
| 3 | `03_ai_self_service_poc.ipynb` | AI self-service platform POC reference |

### 5. Start the BentoML segmentation API

```bash
docker compose -f deployment/docker-compose.yml up --build
```

Available at `http://localhost:3000` — see [API Endpoints](#api-endpoints) below.

### 6. Start the Airflow + MLflow stack

```bash
docker compose -f airflow/docker-compose.yml up --build
```

| Service | URL |
|---------|-----|
| Airflow webserver | `http://localhost:8080` (admin / admin) |
| MLflow UI | `http://localhost:5001` |

### 7. Start the Streamlit chatbot

```bash
streamlit run deployment/streamlit_app.py
```

Opens at `http://localhost:8501`. Ask questions like *"Which product has the highest default rate?"*

### 8. Run tests

```bash
python -m pytest tests/ -v
```

Covers: data loading, preprocessing, feature engineering, segmentation (K=4, silhouette ≥ 0), credit risk model (PR-AUC), and visualization output.

## Project Structure

```
.
├── notebooks/           # 3 Jupyter notebooks (EDA → Segmentation → AI POC)
├── src/                 # Python package
│   ├── data/            # DataLoader, DataPreprocessor
│   ├── features/        # FeatureEngineer
│   ├── models/          # CustomerSegmentation, CreditRiskModel
│   └── viz/             # Plotter (Plotly)
├── deployment/          # BentoML service + Dockerfile + Streamlit app
├── airflow/             # Airflow DAG + Dockerfile + docker-compose
├── proposals/           # Strategic proposals (credit risk, AI platform, deployment)
├── presentation/        # Executive summary PowerPoint
├── models/              # Trained model artifacts (.pkl, .parquet)
├── data/                # Input dataset (gitignored) + SQLite databases
├── tests/               # Smoke tests
├── pyproject.toml       # Package config + dependencies
├── .env                 # Environment variables (gitignored)
└── README.md
```

## API Endpoints

All endpoints are `POST` (BentoML convention). Base URL: `http://localhost:3000`

### `POST /health`

```bash
curl -X POST http://localhost:3000/health
# {"status":"ok","model_loaded":true,"scaler_loaded":true}
```

### `POST /segment_single`

```bash
curl -X POST http://localhost:3000/segment_single \
  -H "Content-Type: application/json" \
  -d '{"input_data": {"account_tenure_days": 365, "risk_score": 0.5, "is_refurbished": 0, "is_payg": 1}}'
```

### `POST /segment_batch`

```bash
curl -X POST http://localhost:3000/segment_batch \
  -H "Content-Type: application/json" \
  -d '{"input_data": {"records": [{"account_tenure_days": 365, "risk_score": 0.5, "is_refurbished": 0, "is_payg": 1}]}}'
```

### `POST /profile`

```bash
curl -X POST http://localhost:3000/profile
```

## Customer Segments

| Segment | Label | Default Rate | Key Trait |
|---------|-------|-------------|-----------|
| 0 | Cash Defaulters | 48.9% | All-cash, highest risk |
| 1 | PAYG Defaulters | 52.1% | Exclusively PAYG, worst profile |
| 2 | Refurbished Buyers | 38.8% | Refurbished units, moderate risk |
| 3 | Healthy Completers | 0% | Zero defaults, ideal customers |

## Key Numbers (verified from source data)

| Metric | Value |
|--------|-------|
| Portfolio default rate | 37.6% |
| Highest write-off product | Water Pump Kit (15.3%) |
| PAYG arrears vs CASH | 13.3% vs 11.2% |
| Gender default gap | Male 40.1% vs Female 36.0% |
| Optimal clusters (silhouette) | K=4 (0.29) |

## Docker Services

| Compose file | Services | Purpose |
|-------------|----------|---------|
| `deployment/docker-compose.yml` | BentoML API (:3000) | Real-time segment prediction |
| `airflow/docker-compose.yml` | Airflow (:8080) + MLflow (:5001) | Weekly model retraining pipeline |
# sunculture
