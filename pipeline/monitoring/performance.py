import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from utils.config import load_config
import datetime
import os

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
    batch_path = cfg["data"]["batch_output"]
    log_path = cfg["monitoring"]["performance_log"]
    features = load_features()

    df = pd.read_csv(batch_path)

    if "score" in df.columns and "predicted_score" in df.columns:
    
        y_true = df["score"]
        y_pred = df["predicted_score"]
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
    else:
        print("No ground truth 'score' column in batch output. Cannot evaluate performance.")
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame([{
        "timestamp": timestamp,
        "r2": r2,
        "mae": mae
    }])

    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        log_df = pd.concat([log_df, new_row], ignore_index=True)
    else:
        log_df = new_row

    log_df.to_csv(log_path, index=False)
    print(f"Performance logged: r2={r2:.4f}, mae={mae:.4f}")

if __name__ == "__main__":
    main()
