from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


from pipeline.training.train import preprocess

MODEL_PATH = "data/models/model_v1.pkl"
FEATURES_PATH = "data/models/features.txt"

app = FastAPI()


class InputData(BaseModel):
    age: int
    gender: str
    state: str
    monthly_income: float
    epf_balance: float
    debt_amount: float
    household_size: int
    has_chronic_disease: bool
    medical_expense_monthly: float
    mental_stress_level: float
    expected_monthly_expense: float
    has_spouse: bool
    num_children: int
    supports_others: bool
    is_supported: bool


def load_features():
    with open(FEATURES_PATH) as f:
        return [line.strip() for line in f]


def align_features(df, features):
    for col in features:
        if col not in df.columns:
            df[col] = 0
    return df[features]


model = joblib.load(MODEL_PATH)
FEATURES = load_features()


@app.get("/")
def read_root():
    return {"message": "Retirement Prediction API"}


@app.post("/predict")
def predict(data: InputData):
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])
    df = preprocess(df)
    df = align_features(df, FEATURES)
    prediction = model.predict(df)[0]
    return {"retirement_readiness_score": round(prediction, 3)}
