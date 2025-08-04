import pandas as pd
from pipeline.training.train import train_model, preprocess
import joblib


def load_feedback_csv(path="data/output/user_feedback.csv"):
    return pd.read_csv(path)


def retrain_model():
    df = load_feedback_csv()

    df = df[df["feedback"].isin(["Very Satisfied", "Satisfied", "Not Satisfied"])]

    df["retirement_readiness_score"] = df.apply(
        lambda row: (
            row["user_score"]
            if row["feedback"] == "Not Satisfied" and not pd.isna(row["user_score"])
            else row["model_score"]
        ),
        axis=1,
    )

    df = df.dropna(subset=["retirement_readiness_score"])

    df = preprocess(df)

    model, features = train_model(df)

    joblib.dump(model, "data/models/model_v1.pkl")
    with open("data/models/features.txt", "w") as f:
        for feat in features:
            f.write(f"{feat}\n")

    print("Retrained and saved new model")


if __name__ == "__main__":
    retrain_model()
