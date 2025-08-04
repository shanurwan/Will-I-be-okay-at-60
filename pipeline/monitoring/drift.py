import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftPreset
from utils.config import load_config
import os

# Load features function
def load_features():
    with open("data/models/features.txt") as f:
        return [line.strip() for line in f]

def align_features(df, features):
    for col in features:
        if col not in df.columns:
            df[col] = 0
    return df[features]

def main():
    cfg = load_config()
    train_data = pd.read_csv(cfg["data"]["input_path"])
    new_data = pd.read_csv(cfg["data"]["batch_input"])
    features = load_features()

    target = cfg["training"]["target"]
    if target in train_data.columns:
        train_data = train_data.drop(columns=[target])
    if target in new_data.columns:
        new_data = new_data.drop(columns=[target])

    train_data = align_features(train_data, features)
    new_data = align_features(new_data, features)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=train_data, current_data=new_data)

    output_path = cfg["monitoring"]["drift_report"]
    report.save_html(output_path)
    print(f"Drift report saved to {output_path}")

if __name__ == "__main__":
    main()
