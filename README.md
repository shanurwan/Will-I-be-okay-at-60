# Will I Be Okay at 60?

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![CI](https://github.com/shanurwan/Will-I-be-okay-at-60/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

This is a work-in-progress public data tool built to grow over time. To the author knowledge, as of 4 August 2025 this is Malaysian first open source retirement readiness machine learning decision support system.


Check your retirement readiness [here](https://will-i-be-okay-at-60-m5cpstcz8v2gdrvxk8vrcn.streamlit.app/)

---


## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Data](#data)
3. [Project Goals](#project-goals)
4. [Architecture & Structure](#architecture--structure)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Usage](#usage)

   * [Training](#training)
   * [Evaluation](#evaluation)
   * [Serving](#serving)
   * [Batch Inference](#batch-inference)
8. [Monitoring & Automation](#monitoring--automation)
9. [Testing & CI](#testing--ci)
10. [Contributing](#contributing)
11. [License](#license)

---

## Problem Statement

As Malaysia’s aging population grows, many approach retirement without sufficient planning. This pipeline provides a decision-support tool to forecast a retirement readiness score at age 60, helping individuals and stakeholders identify gaps in financial, health, and social preparedness.

## Data

The dataset integrates multiple domains:

* **Financial**: Monthly income, EPF balance, debt amount
* **Medical**: Chronic disease flag, monthly medical expenses
* **Psychological**: Self-reported mental stress level
* **Demographic & Family**: Age, gender, state, household size, family dependency
* **Lifestyle**: Expected monthly living expenses

> **Note:** Raw CSVs reside under `data/input/`. Preprocessed features and trained model artifacts output to `data/models/`.

Data Documentation = [Click here](https://github.com/shanurwan/Malaysian-Retirement-Dataset)

## Project Goals

* **Predict** retirement readiness score
* **Automate** end-to-end pipeline: training → evaluation → serving → monitoring → scheduling
* **Monitor** model drift and performance over time
* **Iterate** via feedback loop to improve predictions

## Architecture & Structure

```
Will-I-be-okay-at-60/
├── README.md
├── LICENSE
├── Makefile
├── .env.sample                 # local-only envs (DATABASE_URL, LOG_LEVEL, etc.)
├── requirements/               # split deps to keep Lambda lean
│   ├── base.txt
│   ├── api.txt                 # FastAPI + Mangum + prometheus_client
│   └── dev.txt                 # pytest, black, isort, mypy, etc.
├── pipeline/                   # your app code stays here
│   ├── settings.py             # Pydantic settings (reads env vars)
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── migrations/         # Alembic (schema as code)
│   │       ├── env.py
│   │       └── versions/
│   ├── serving/
│   │   ├── api.py              # FastAPI (GET /health, /metrics; POST /predict)
│   │   ├── api_lambda.py       # Mangum handler: `handler` for Lambda URL
│   │   ├── batch.py            # CLI batch inference (local/dev)
│   │   └── batch_lambda.py     # Batch/drift/perf handlers (EventBridge)
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   └── monitoring/
│       ├── drift.py
│       └── performance.py
├── scripts/
│   ├── sheets_to_pg.py         # one-off ETL: Google Sheets → Postgres (Neon)
│   ├── create_tables.py        # bootstrap DB (or use Alembic)
│   └── load_sample_data.py
├── web/                        # static web UI (free on GitHub Pages)
│   ├── index.html              # calls Lambda Function URL for /predict
│   ├── app.js
│   └── styles.css
├── docs/
│   ├── architecture.md         # diagrams + flow (serverless, zero-cost guardrails)
│   ├── api.md                  # request/response schema & examples
│   ├── runbook.md              # deploy, rollback, debug, smoke tests
│   └── cost-guardrails.md      # what to avoid to stay at $0
├── infra/
│   └── terraform/              # infra-as-code (replaces local/production YAML)
│       ├── providers.tf
│       ├── variables.tf
│       ├── locals.tf
│       ├── outputs.tf
│       ├── main.tf             # wires modules together
│       ├── lambda/             # Lambda + Function URL + logs
│       │   └── lambda.tf
│       ├── scheduler/          # EventBridge schedules (batch, drift, perf)
│       │   └── scheduler.tf
│       ├── ssm/                # optional: free SSM Standard Parameters for env
│       │   └── params.tf
│       └── neon/               # Neon project/db via provider (or doc placeholders)
│           └── neon.tf
├── .github/
│   └── workflows/
│       ├── ci.yml              # lint, type-check, tests, dep scan
│       └── cd.yml              # package zip, deploy via Terraform (OIDC) or gh-actions
├── tests/
│   ├── unit/
│   │   └── test_api.py
│   └── integration/
│       └── test_db.py
└── ops/
    ├── grafana/                # JSON dashboards (latency, RPS, errors)
    └── alerts/                 # PromQL notes / SLOs (for future)

```

## Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/shanurwan/Will-I-be-okay-at-60.git
   cd Will-I-be-okay-at-60
   ```
2. Create & activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate   # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Configure paths, thresholds, and credentials in `config/local.yaml` for local development. For production-like runs, adjust `config/production.yaml` accordingly.

## Usage

### Training

```bash
python pipeline/training/train.py --config config/local.yaml
```

Logs & metrics (loss, RMSE) will be stored in MLflow (`mlruns/`).

### Evaluation

```bash
python pipeline/training/evaluate.py --config config/local.yaml
```

### Serving

Run the FastAPI application for real-time prediction:

```bash
uvicorn pipeline/serving/api:app --reload --port 8000
```

Endpoints:

* `POST /predict` → Predict retirement readiness
* `GET /health` → Health check

### Batch Inference

Schedule bulk predictions via:

```bash
python pipeline/serving/batch.py --input data/input/new_users.csv --output data/models/predictions.csv
```

## Monitoring & Automation

* **Drift Detection**: Use `pipeline/monitoring/drift.py` (Evidently) to monitor feature/data drift.
* **Performance Monitoring**: `pipeline/monitoring/performance.py` logs ongoing metrics.
* **Orchestration**: `orchestration/dag.py` defines the end-to-end Prefect workflow; schedule using `orchestration/scheduler.py` or GitHub Actions cron.
* **Retraining**: Automate retraining on drift detection or at scheduled intervals (e.g., monthly via GitHub Actions).

## Testing & CI

Run tests locally:

```bash
pytest --maxfail=1 --disable-warnings -q
flake8
```

Continuous Integration is configured under `.github/workflows/ci.yml` to run tests and linting on each push.

## Contributing

Contributions are welcome! Please:

1. Fork the repo
2. Create a feature branch
3. Open a pull request

## License

This project is licensed under the MIT License.

## Contact

Email = wannurshafiqah18@gmail.com
