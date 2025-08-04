import os
import json
import joblib
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score
from pipeline.training.train import train_model, preprocess

def load_feedback_from_gsheet(sheet_name="Feedback", worksheet_index=0):
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials

    creds_dict = json.loads(os.environ["GCP_SA_KEY"])

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
 

    client = gspread.authorize(creds)
    sheet = client.open(sheet_name).get_worksheet(worksheet_index)

    print("Connected to Google Sheet.")
    records = sheet.get_all_records()
    return pd.DataFrame(records)


def log_performance(y_true, y_pred, model_version="latest", log_path="data/metrics/performance_log.csv"):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "mae": mae,
        "r2": r2,
        "n_samples": len(y_true),
    }

    log_df = pd.DataFrame([log_entry])
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, mode="w", header=True, index=False)

    print(f"Performance: MAE={mae:.4f}, R²={r2:.4f}")


def retrain_model():
    df = load_feedback_from_gsheet()

    df = df[df["feedback"].isin(["Very Satisfied", "Satisfied", "Not Satisfied"])]

    df["retirement_readiness_score"] = df.apply(
        lambda row: row["user_score"] if row["feedback"] == "Not Satisfied" and not pd.isna(row["user_score"])
        else row["model_score"], axis=1
    )


    df = df.dropna(subset=["retirement_readiness_score"])

    y_true = df["retirement_readiness_score"].copy()

    df = preprocess(df)
    model, features = train_model(df)
    y_pred = model.predict(df)

   
    log_performance(y_true, y_pred)

    os.makedirs("data/models", exist_ok=True)
    joblib.dump(model, "data/models/model_v1.pkl")
    with open("data/models/features.txt", "w") as f:
        for feat in features:
            f.write(f"{feat}\n")

    print("Retrained model saved.")

if __name__ == "__main__":
    retrain_model()
