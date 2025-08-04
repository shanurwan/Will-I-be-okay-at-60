import pandas as pd
import joblib
from pipeline.training.train import preprocess, TARGET

# Constants
MODEL_PATH = "data/models/model_v1.pkl"
INPUT_PATH = "data/input/retire.csv"
OUTPUT_PATH = "data/output/predictions.csv"


def main():
    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Loading input data...")
    data = pd.read_csv(INPUT_PATH)

    print("Preprocessing...")
    data_processed = preprocess(data)

    if TARGET in data_processed.columns:
        X = data_processed.drop(columns=[TARGET])
    else:
        X = data_processed

    print("Making predictions...")
    predictions = model.predict(X)

    print("Saving results...")
    data["predicted_score"] = predictions
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Predictions saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
