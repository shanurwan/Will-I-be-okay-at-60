import streamlit as st
import pandas as pd
import joblib
import os
import csv
from pipeline.training.train import preprocess

MODEL_PATH = "data/models/model_v1.pkl"
FEATURES_PATH = "data/models/features.txt"
FEEDBACK_PATH = "data/output/user_feedback.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_features():
    with open(FEATURES_PATH) as f:
        return [line.strip() for line in f]


def align_features(df, features):
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]
    return df


def readiness_status(score):
    if score >= 0.7:
        return "READY for retirement", (
            "Your score is high. This usually means you have healthy income, "
            "good savings, and manageable expenses/debt."
        )
    else:
        return "NOT READY yet", (
            "Your score is low. Common reasons: insufficient savings, high debt, "
            "high expenses, or medical/household challenges."
        )


def save_feedback(input_dict, model_score, user_score, feedback):
    data = {
        **input_dict,
        "model_score": model_score,
        "user_score": user_score,
        "feedback": feedback,
    }
    file_exists = os.path.exists(FEEDBACK_PATH)
    with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


st.title("Malaysian Retirement Readiness Predictor")
st.header("Predict for a single user")


with st.form("single_form"):
    age = st.number_input("Age", 40, 100, 55)
    gender = st.selectbox("Gender", ["Male", "Female"])
    state = st.selectbox(
        "State",
        [
            "Johor",
            "Kedah",
            "Kelantan",
            "Melaka",
            "Negeri Sembilan",
            "Pahang",
            "Penang",
            "Perak",
            "Perlis",
            "Sabah",
            "Sarawak",
            "Selangor",
            "Terengganu",
            "W.P. Kuala Lumpur",
            "W.P. Labuan",
            "W.P. Putrajaya",
        ],
    )
    monthly_income = st.number_input("Monthly Income", 0, 100_000, 3000)
    epf_balance = st.number_input("EPF Balance", 0, 1_000_000, 40_000)
    debt_amount = st.number_input("Debt Amount", 0, 100_000, 0)
    household_size = st.slider("Household Size", 1, 10, 3)
    has_chronic_disease = st.checkbox("Has Chronic Disease?")
    medical_expense_monthly = st.number_input("Monthly Medical Expense", 0, 10_000, 0)
    mental_stress_level = st.slider("Mental Stress Level", 0.0, 1.0, 0.5)
    expected_monthly_expense = st.number_input(
        "Expected Monthly Expense", 0, 20_000, 1500
    )
    has_spouse = st.checkbox("Has Spouse?")
    num_children = st.slider("Number of Children", 0, 10, 1)
    supports_others = st.checkbox("Supports Others?")
    is_supported = st.checkbox("Is Supported by Others?")

    submitted = st.form_submit_button("Predict Retirement Readiness")
    if submitted:
        input_dict = {
            "age": age,
            "gender": gender,
            "state": state,
            "monthly_income": monthly_income,
            "epf_balance": epf_balance,
            "debt_amount": debt_amount,
            "household_size": household_size,
            "has_chronic_disease": has_chronic_disease,
            "medical_expense_monthly": medical_expense_monthly,
            "mental_stress_level": mental_stress_level,
            "expected_monthly_expense": expected_monthly_expense,
            "has_spouse": has_spouse,
            "num_children": num_children,
            "supports_others": supports_others,
            "is_supported": is_supported,
        }
        df = pd.DataFrame([input_dict])
        df = preprocess(df)
        features = load_features()
        df = align_features(df, features)
        model = load_model()
        score = model.predict(df)[0]
        score_rounded = round(score, 3)

        st.session_state["input_dict"] = input_dict
        st.session_state["score_rounded"] = score_rounded
        st.session_state["shown_prediction"] = True


if st.session_state.get("shown_prediction", False):
    input_dict = st.session_state["input_dict"]
    score_rounded = st.session_state["score_rounded"]
    status, reason = readiness_status(score_rounded)
    st.success(f"Estimated Retirement Readiness Score: {score_rounded}")
    st.info(f"{status}\n\n{reason}")

    st.markdown("### Was this prediction accurate for you?")
    with st.form("feedback_form"):
        feedback = st.radio(
            "How satisfied are you with the prediction?",
            ["Very Satisfied", "Satisfied", "Not Satisfied"],
            key="feedback_radio",
        )
        user_score = None
        if feedback == "Not Satisfied":
            user_score = st.number_input(
                "What do you think your real readiness score should be?",
                min_value=0.0,
                max_value=1.0,
                value=score_rounded,
                step=0.01,
                key="user_score_input",
            )
        feedback_submitted = st.form_submit_button("Submit Feedback")
        if feedback_submitted:
            save_feedback(input_dict, score_rounded, user_score, feedback)
            st.success("Thank you for your feedback!")
            st.session_state["shown_prediction"] = False




st.caption("by Wan Nur Shafiqah, 2025")
