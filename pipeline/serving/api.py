from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pipeline.training.train import preprocess, FEATURES

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


MODEL_PATH = "data/models/model_v1.pkl"
model = joblib.load(MODEL_PATH)


@app.get("/")
def read_root():
    return {"message": "Retirement Prediction API"}


@app.post("/predict")
def predict(data: InputData):
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])
    df = preprocess(df)
    prediction = model.predict(df[FEATURES])[0]
    return {"retirement_readiness_score": round(prediction, 3)}
