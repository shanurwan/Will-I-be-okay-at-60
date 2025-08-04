import streamlit as st
import joblib
import gspread
from datetime import datetime


MODEL_PATH = "data/models/model_v1.pkl"
FEATURES_PATH = "data/models/features.txt"


@st.cache_resource
def get_sheet():
    # 1) Grab and normalize the creds dict
    creds_info = dict(st.secrets["gcp_service_account"])
    creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
    # 2) Let gspread build the client directly
    client = gspread.service_account_from_dict(creds_info)
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
    # … your inputs as before …
    if st.form_submit_button("Predict Retirement Readiness"):
        # … prediction logic …
        st.session_state["shown_prediction"] = True

if st.session_state.get("shown_prediction", False):
    # … display results …
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
                value=st.session_state["score_rounded"],
                step=0.01,
                key="user_score_input",
            )
        if st.form_submit_button("Submit Feedback"):
            save_feedback(
                st.session_state["input_dict"],
                st.session_state["score_rounded"],
                user_score,
                feedback,
            )
            st.success("Thank you for your feedback! 🎉")
            st.session_state["shown_prediction"] = False

st.caption("by Wan Nur Shafiqah, 2025")
