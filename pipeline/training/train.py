import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import mlflow
import joblib
import os
from utils.config import load_config

TARGET = "score"

def preprocess(df):
    df = df.copy()
    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
    df["has_chronic_disease"] = df["has_chronic_disease"].astype(int)
    df["has_spouse"] = df["has_spouse"].astype(int)
    df["supports_others"] = df["supports_others"].astype(int)
    df["is_supported"] = df["is_supported"].astype(int)
    df = pd.get_dummies(df, columns=["state"], drop_first=True)
    return df

def train_model(X_train, y_train, n_estimators):
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

def save_features(features, path):
    with open(path, "w") as f:
        for col in features:
            f.write(f"{col}\n")

if __name__ == "__main__":

    cfg = load_config("config/local.yaml")

    data_path = cfg["data"]["input_path"]
    model_path = cfg["data"]["model_path"]
    features_path = os.path.join(os.path.dirname(model_path), "features.txt")
    test_size = cfg["training"]["test_size"]
    random_state = cfg["training"]["random_state"]
    n_estimators = cfg["training"]["n_estimators"]
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    experiment_name = cfg["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    df = pd.read_csv(data_path)
    df = preprocess(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    FEATURES = X.columns.tolist() 

    save_features(FEATURES, features_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    with mlflow.start_run():
       
        model = train_model(X_train, y_train, n_estimators)
        r2, mae = evaluate_model(model, X_test, y_test)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(model, "model")
        save_model(model, model_path)

    print(f"Training complete! Model saved to: {model_path}")
    print(f"Feature list saved to: {features_path}")
    print(f"R²: {r2:.4f}, MAE: {mae:.4f}")
