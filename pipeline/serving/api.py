
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import mlflow.sklearn
import os

from pipeline.training.train import preprocess, FEATURES


# Configuration

MODEL_PATH = "data/models/model_v1.pkl"  


# FastAPI setup

app = FastAPI(title="Retirement Readiness API", version="1.0")


# Input schema

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



# Load model

model = joblib.load(MODEL_PATH)



# API Endpoint

@app.post("/predict")
def predict(data: InputData):
    
    df = pd.DataFrame([data.dict()])
    df = preprocess(df)

   
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURES] 

   
    score = model.predict(df)[0]
    return {"retirement_score": round(float(score), 4)}
