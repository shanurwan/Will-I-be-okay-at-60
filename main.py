import streamlit as st
import pandas as pd
import joblib
import gspread
import tempfile
import json
from google.oauth2.service_account import Credentials
from datetime import datetime
from pipeline.training.train import preprocess

MODEL_PATH = "data/models/model_v1.pkl"
FEATURES_PATH = "data/models/features.txt"


@st.cache_resource
def get_sheet():

    creds_info = dict(st.secrets["gcp_service_account"])
    creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
        json.dump(creds_info, tmp)
        tmp_path = tmp.name

    creds = Credentials.from_service_account_file(
        tmp_path, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(creds)
    return client.open_by_key(st.secrets["sheet_id"]).sheet1


def save_feedback(input_dict, model_score, user_score, feedback):
    sheet = get_sheet()
    row = [
        input_dict["age"],
        input_dict["gender"],
        input_dict["state"],
        input_dict["monthly_income"],
        input_dict["epf_balance"],
        input_dict["debt_amount"],
        input_dict["household_size"],
        int(input_dict["has_chronic_disease"]),
        input_dict["medical_expense_monthly"],
        input_dict["mental_stress_level"],
        input_dict["expected_monthly_expense"],
        int(input_dict["has_spouse"]),
        input_dict["num_children"],
        int(input_dict["supports_others"]),
        int(input_dict["is_supported"]),
        model_score,
        user_score if user_score is not None else "",
        feedback,
        datetime.utcnow().isoformat(),
    ]
    sheet.append_row(row, value_input_option="USER_ENTERED")


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
    return df[features]


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

    if st.form_submit_button("Predict Retirement Readiness"):
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
                0.0,
                1.0,
                score_rounded,
                0.01,
                key="user_score_input",
            )
        if st.form_submit_button("Submit Feedback"):
            save_feedback(input_dict, score_rounded, user_score, feedback)
            st.success("Thank you for your feedback!")
            st.session_state["shown_prediction"] = False

st.caption("by Wan Nur Shafiqah, 2025")
