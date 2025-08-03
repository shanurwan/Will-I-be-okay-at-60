import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.training.train import train_model


from pipeline.training.train import train_model
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def test_train_model_returns_model():
    # dummy data
    X = pd.DataFrame(
        {
            "age": [50, 55],
            "gender": [0, 1],
            "state_Selangor": [1, 0],  # dummy one-hot
            "monthly_income": [5000, 7000],
            "epf_balance": [40000, 60000],
            "debt_amount": [1000, 2000],
            "household_size": [3, 4],
            "has_chronic_disease": [0, 1],
            "medical_expense_monthly": [200, 300],
            "mental_stress_level": [0.5, 0.6],
            "expected_monthly_expense": [1500, 1600],
            "has_spouse": [1, 0],
            "num_children": [1, 2],
            "supports_others": [0, 1],
            "is_supported": [1, 0],
        }
    )
    y = pd.Series([0.6, 0.8])

    model = train_model(X, y)
    assert isinstance(model, RandomForestRegressor)
    assert hasattr(model, "predict")
    assert model.predict([X.iloc[0]]).shape == (1,)


def test_dummy_pass():
    assert 1 == 1
