import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import mlflow
import joblib

FEATURES = [
    "age", "gender", "state", "monthly_income", "epf_balance", "debt_amount",
    "household_size", "has_chronic_disease", "medical_expense_monthly",
    "mental_stress_level", "expected_monthly_expense", "has_spouse",
    "num_children", "supports_others", "is_supported"
]

TARGET = "score"

def preprocess(df):
    data = data.copy()
    # Encode categoricals
    data["gender"] = df["gender"].map({"Male": 0, "Female": 1})
    data["has_chronic_disease"] = data["has_chronic_disease"].astype(int)
    data["has_spouse"] = df["has_spouse"].astype(int)
    data["supports_others"] = data["supports_others"].astype(int)
    data["is_supported"] = data["is_supported"].astype(int)

    # One-hot encode state
    data = pd.get_dummies(data, columns=["state"], drop_first=True)

    return data

def train_model(X_train, y_train, n_estimators=100):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return r2, mae

def save_model(model, path):
    joblib.dump(model, path)

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("retirement_model")

    # Load + preprocess
    data = data.read_csv("data/input/retirement.csv")
    data = preprocess(data)

    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = train_model(X_train, y_train)
        r2, mae = evaluate_model(model, X_test, y_test)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(model, "model")

        save_model(model, "data/models/model_v1.pkl")
