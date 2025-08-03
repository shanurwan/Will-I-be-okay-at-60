import pandas as pd
import joblib
import mlflow
from sklearn.metrics import r2_score, mean_absolute_error
from pipeline.training.train import preprocess, FEATURES, TARGET


def load_model(path: str):
    return joblib.load(path)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return r2, mae, y_pred


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("retirement_model")

    # Load data and model
    data = pd.read_csv("data/input/retirement.csv")
    data = preprocess(data)
    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    # For simplicity, reuse same train/test split
    X_test = X.sample(frac=0.2, random_state=42)
    y_test = y.loc[X_test.index]

    model = load_model("data/models/model_v1.pkl")

    with mlflow.start_run(run_name="evaluate_model"):
        r2, mae, _ = evaluate_model(model, X_test, y_test)
        mlflow.log_metric("eval_r2", r2)
        mlflow.log_metric("eval_mae", mae)
