import pandas as pd
import joblib
from pipeline.training.train import preprocess, TARGET


MODEL_PATH = "data/models/model_v1.pkl"
FEATURES_PATH = "data/models/features.txt"
INPUT_PATH = "data/input/retire.csv"
OUTPUT_PATH = "data/output/predictions.csv"


def load_features():
    with open(FEATURES_PATH) as f:
        return [line.strip() for line in f]


def align_features(df, features):
    for col in features:
        if col not in df.columns:
            df[col] = 0
    return df[features]


def main():
    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Loading input data...")
    data = pd.read_csv(INPUT_PATH)

    print("Preprocessing...")
    data_processed = preprocess(data)

    features = load_features()
    if TARGET in data_processed.columns:
        X = data_processed.drop(columns=[TARGET])
    else:
        X = data_processed

    X = align_features(X, features)

    print("Making predictions...")
    predictions = model.predict(X)

    print("Saving results...")
    data["predicted_score"] = predictions
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Predictions saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
