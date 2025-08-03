

import pandas as pd
import joblib
import mlflow
from sklearn.metrics import r2_score, mean_absolute_error
from pipeline.training.train import preprocess, FEATURES, TARGET
from utils.config import load_config

def load_model(path: str):
    return joblib.load(path)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return r2, mae, y_pred

if __name__ == "__main__":
    
    cfg = load_config("config/local.yaml") 

    
    data_path = cfg["data"]["input_path"]
    model_path = cfg["data"]["model_path"]
    test_size = cfg["training"]["test_size"]
    random_state = cfg["training"]["random_state"]
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    experiment_name = cfg["mlflow"]["experiment_name"]

   
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    
    df = pd.read_csv(data_path)
    df = preprocess(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

   
    X_test = X.sample(frac=test_size, random_state=random_state)
    y_test = y.loc[X_test.index]

    model = load_model(model_path)

    with mlflow.start_run(run_name="evaluate_model"):
        r2, mae, _ = evaluate_model(model, X_test, y_test)
        mlflow.log_metric("eval_r2", r2)
        mlflow.log_metric("eval_mae", mae)

        print(f"Evaluation R²: {r2:.4f}")
        print(f"Evaluation MAE: {mae:.4f}")
