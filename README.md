# Will I Be Okay at 60? 

A production-ready ML pipeline that predicts retirement readiness score for Malaysians using financial, medical, and psychological features. 

---

## Data

Due data inavailability, the initial stage used simulated data grounded in real world statistic

[Click here to see more](https://github.com/shanurwan/Malaysian-Retirement-Dataset)

---

## Project Goals

- Predict retirement readiness score
- Run reproducible, end-to-end ML pipeline
- Include training, serving, monitoring, scheduling 
- Improve prediction quality over time using feedback loop

---

## Project Structure

```text
retirement_project/
├── data/
│   ├── input/               # input data
│   └── models/           
│
├── pipeline/
│   ├── training/
│   │   ├── train.py       # Model training script
│   │   └── evaluate.py    # Evaluation and metric logging
│   │
│   ├── serving/
│   │   ├── api.py         # FastAPI app for real-time predictions
│   │   └── batch.py       # CLI for batch inference jobs
│   │
│   └── monitoring/
│       ├── drift.py       # Data drift detection (via Evidently)
│       └── performance.py # Monitoring model performance over time
│
├── orchestration/
│   ├── dag.py             # DAG defining ML workflow
│   └── scheduler.py       # Cron-based task scheduler
│
├── config/
│   ├── local.yaml         # Local run configs (paths, thresholds, etc.)
│   └── production.yaml    # Simulated production configs
│
├── tests/
│   └── test_pipeline.py   # Unit & integration tests
│
├── requirements.txt       # Python dependencies
└── README.md              
```
