import streamlit as st
import pandas as pd
import joblib
import requests
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
from pipeline.training.train import preprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "data/models/model_v1.pkl"
FEATURES_PATH = "data/models/features.txt"
READINESS_THRESHOLD = 0.7

# Styling
st.set_page_config(
    page_title="Malaysian Retirement Readiness Predictor",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .prediction-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .recommendation-box {
        background-color: #e8f4fd;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Cached, top-level loader
# ---------------------------
@st.cache_resource
def load_model_components() -> Tuple[Optional[Any], Optional[List[str]], Optional[Any]]:
    """Load model, feature list, and SHAP explainer (cached across reruns)."""
    try:
        model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
            features = [line.strip() for line in f if line.strip()]

        # Initialize SHAP explainer (tree models). Adjust if using other model types.
        explainer = shap.TreeExplainer(model)

        logger.info("Model and SHAP explainer loaded successfully")
        return model, features, explainer

    except Exception as e:
        logger.error(f"Error loading model components: {e}")
        st.error(f"Failed to load model: {e}")
        return None, None, None


class RetirementPredictor:
    """
    Professional retirement readiness prediction system with SHAP explanations.
    """

    def __init__(self):
        self.model, self.features, self.explainer = load_model_components()

    def align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align DataFrame features with model requirements."""
        if not self.features:
            return df
        for col in self.features:
            if col not in df.columns:
                df[col] = 0
        # Keep only the features expected by the model (order matters)
        return df[self.features]

    def predict(self, input_data: Dict[str, Any]) -> Tuple[float, np.ndarray]:
        """
        Make prediction and return score with SHAP values.

        Returns:
            (prediction_score, shap_values_for_first_row)
        """
        try:
            # Convert to DataFrame and preprocess
            df = pd.DataFrame([input_data])
            df = preprocess(df)
            df = self.align_features(df)

            # Make prediction
            score = float(self.model.predict(df)[0])

            # Calculate SHAP values; handle different return shapes from SHAP
            raw_shap = self.explainer.shap_values(df)

            # If SHAP returns a list (e.g., classifier), pick the positive class if available
            if isinstance(raw_shap, list):
                shap_arr = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
            else:
                shap_arr = raw_shap

            shap_values_row = np.array(shap_arr[0])

            return score, shap_values_row

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise e

    def get_readiness_status(self, score: float) -> Tuple[str, str, str]:
        """
        Get readiness status with detailed explanation.

        Returns:
            (status, message, color)
        """
        if score >= READINESS_THRESHOLD:
            return (
                "READY for Retirement",
                "Congratulations! Your retirement readiness score indicates strong financial preparation. "
                "You have demonstrated good financial health with adequate savings, manageable expenses, and stable income.",
                "success"
            )
        else:
            return (
                "NEEDS IMPROVEMENT",
                "Your retirement readiness score suggests areas for improvement. "
                "Focus on increasing savings, reducing debt, and optimizing your financial strategy.",
                "warning"
            )

    def generate_personalized_recommendations(self, shap_values: np.ndarray, input_data: Dict[str, Any]) -> List[str]:
        """
        Generate personalized recommendations based on SHAP values.
        """
        recommendations: List[str] = []

        if not self.features or shap_values is None or len(shap_values) != len(self.features):
            # Fallback generic guidance if SHAP/feature alignment is off
            return [
                " **Regular Review**: Monitor and adjust your retirement plan annually",
                " **Professional Advice**: Consider consulting with a certified financial planner",
                " **Emergency Fund**: Maintain 6-12 months of expenses in emergency savings"
            ]

        # Feature importance (absolute SHAP values)
        feature_importance = dict(zip(self.features, np.abs(shap_values)))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:5]

        shap_map = dict(zip(self.features, shap_values))
        for feature, _importance in top_features:
            s_val = shap_map.get(feature, 0.0)

            if feature == 'epf_balance' and s_val < 0:
                recommendations.append(
                    " **Increase EPF Contributions**: Your EPF balance significantly impacts your readiness. "
                    "Consider increasing voluntary contributions or exploring EPF investment schemes."
                )
            elif feature == 'monthly_income' and s_val < 0:
                recommendations.append(
                    " **Boost Monthly Income**: Consider skill development, career advancement, or additional income streams "
                    "to improve your financial position."
                )
            elif feature == 'debt_amount' and s_val < 0:
                recommendations.append(
                    " **Debt Management**: High debt is affecting your readiness. Create a debt repayment plan "
                    "and consider debt consolidation options."
                )
            elif feature == 'expected_monthly_expense' and s_val < 0:
                recommendations.append(
                    " **Expense Planning**: Review and optimize your expected retirement expenses. "
                    "Consider lifestyle adjustments or relocating to areas with lower living costs."
                )
            elif feature == 'medical_expense_monthly' and s_val < 0:
                recommendations.append(
                    " **Healthcare Planning**: High medical expenses impact readiness. "
                    "Consider comprehensive health insurance and preventive healthcare measures."
                )
            elif feature == 'age' and s_val < 0:
                recommendations.append(
                    " **Time-Sensitive Planning**: Given your age, focus on aggressive savings and investment strategies "
                    "to maximize your remaining working years."
                )

        # Ensure at least a few actionable tips
        if len(recommendations) < 3:
            recommendations.extend([
                " **Regular Review**: Monitor and adjust your retirement plan annually",
                " **Professional Advice**: Consider consulting with a certified financial planner",
                " **Emergency Fund**: Maintain 6-12 months of expenses in emergency savings",
            ])

        return recommendations[:5]


class FeedbackManager:
    """Handle user feedback submission."""

    @staticmethod
    def save_feedback(input_dict: Dict[str, Any], model_score: float,
                      user_score: Optional[float], feedback: str) -> bool:
        """
        Save user feedback to external service.
        """
        payload = {
            **input_dict,
            "model_score": model_score,
            "user_score": user_score,
            "feedback": feedback,
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            response = requests.post(
                st.secrets["feedback_webhook"],
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return True

        except requests.exceptions.ReadTimeout:
            st.warning(
                " Feedback submission timed out. Your feedback may still have been recorded. "
                "Please check the Google Sheet or try again."
            )
            return False

        except requests.exceptions.RequestException as e:
            st.error(f" Failed to submit feedback: {e}")
            return False


def create_shap_visualization(shap_values: np.ndarray, features: List[str], input_data: Dict[str, Any]):
    """Create SHAP waterfall-style bar plot."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        shap_dict = dict(zip(features, shap_values))
        # Sort by absolute SHAP value
        sorted_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_items[:10]  # Show top 10 features

        feature_names = [item[0] for item in top_features]
        shap_vals = [item[1] for item in top_features]

        # Horizontal bar plot
        colors = ['red' if val < 0 else 'green' for val in shap_vals]
        bars = ax.barh(range(len(feature_names)), shap_vals, color=colors, alpha=0.7)

        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title('Feature Impact on Retirement Readiness Score')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, shap_vals)):
            ax.text(val + (0.001 if val > 0 else -0.001), i, f'{val:.3f}',
                    va='center', ha='left' if val > 0 else 'right')

        plt.tight_layout()
        return fig

    except Exception as e:
        logger.error(f"Error creating SHAP visualization: {e}")
        return None


def main():
    """Main application function."""

    # Header
    st.markdown('<h1 class="main-header"> Malaysian Retirement Readiness Predictor</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Assess your retirement preparedness with AI-powered insights and personalized recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize predictor
    predictor = RetirementPredictor()

    if predictor.model is None:
        st.error("Unable to load prediction model. Please contact support.")
        return

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Personal Information")

        with st.form("prediction_form"):
            # Personal details
            st.subheader("Basic Details")
            col_a, col_b = st.columns(2)

            with col_a:
                age = st.number_input("Age", min_value=40, max_value=100, value=55,
                                      help="Your current age")
                gender = st.selectbox("Gender", ["Male", "Female"])
                state = st.selectbox("State", [
                    "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan",
                    "Pahang", "Penang", "Perak", "Perlis", "Sabah", "Sarawak",
                    "Selangor", "Terengganu", "W.P. Kuala Lumpur", "W.P. Labuan", "W.P. Putrajaya"
                ])

            with col_b:
                household_size = st.slider("Household Size", min_value=1, max_value=10, value=3)
                has_spouse = st.checkbox("Married/Has Spouse")
                num_children = st.slider("Number of Children", min_value=0, max_value=10, value=1)

            # Financial details
            st.subheader("Financial Information")
            col_c, col_d = st.columns(2)

            with col_c:
                monthly_income = st.number_input("Monthly Income (RM)", min_value=0,
                                                 max_value=100_000, value=3000,
                                                 help="Your current monthly income")
                epf_balance = st.number_input("EPF Balance (RM)", min_value=0,
                                              max_value=1_000_000, value=40_000,
                                              help="Your current EPF account balance")
                debt_amount = st.number_input("Total Debt Amount (RM)", min_value=0,
                                              max_value=100_000, value=0,
                                              help="Total outstanding debt")

            with col_d:
                expected_monthly_expense = st.number_input("Expected Monthly Retirement Expense (RM)",
                                                           min_value=0, max_value=20_000, value=1500,
                                                           help="Estimated monthly expenses during retirement")
                medical_expense_monthly = st.number_input("Monthly Medical Expense (RM)",
                                                          min_value=0, max_value=10_000, value=0)
                mental_stress_level = st.slider("Mental Stress Level", min_value=0.0,
                                                max_value=1.0, value=0.5, step=0.1,
                                                help="Rate your stress level (0=no stress, 1=high stress)")

            # Health and support
            st.subheader("Health & Support")
            col_e, col_f = st.columns(2)

            with col_e:
                has_chronic_disease = st.checkbox("Has Chronic Disease",
                                                  help="Do you have any chronic medical conditions?")
                supports_others = st.checkbox("Supports Others Financially",
                                              help="Do you provide financial support to others?")

            with col_f:
                is_supported = st.checkbox("Receives Financial Support",
                                           help="Do you receive financial support from others?")

            submitted = st.form_submit_button("🔮 Predict Retirement Readiness",
                                              type="primary", use_container_width=True)

        # Process prediction
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

            try:
                score, shap_values = predictor.predict(input_dict)
                score_rounded = round(score, 3)

                # Store in session state
                st.session_state.update({
                    "input_dict": input_dict,
                    "score_rounded": score_rounded,
                    "shap_values": shap_values,
                    "shown_prediction": True
                })

                st.rerun()

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with col2:
        st.header(" Information")
        st.info("""
        **How it works:**

        1. Fill in your personal and financial details
        2. Get your retirement readiness score
        3. Receive personalized recommendations
        4. View detailed feature analysis

        **Score Interpretation:**
        - 0.7 - 1.0: Ready for retirement
        - 0.0 - 0.7: Needs improvement
        """)

    # Display results if prediction has been made
    if st.session_state.get("shown_prediction", False):
        display_results(predictor)


def display_results(predictor: RetirementPredictor):
    """Display prediction results and recommendations."""

    input_dict = st.session_state.get("input_dict")
    score_rounded = st.session_state.get("score_rounded")
    shap_values = st.session_state.get("shap_values")

    st.markdown("---")
    st.header("Prediction Results")

    # Main prediction display
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(f"""
        <div class="prediction-container">
            <h2 style="text-align: center; margin-bottom: 20px;">Retirement Readiness Score</h2>
            <h1 style="text-align: center; font-size: 4rem; color: {'#28a745' if score_rounded >= READINESS_THRESHOLD else '#ffc107'};">
                {score_rounded}
            </h1>
        </div>
        """, unsafe_allow_html=True)

    # Status and explanation
    status, message, color = predictor.get_readiness_status(score_rounded)

    if color == "success":
        st.success(f"**{status}**\n\n{message}")
    else:
        st.warning(f"**{status}**\n\n{message}")

    # Personalized recommendations
    st.header("Personalized Recommendations")
    recommendations = predictor.generate_personalized_recommendations(shap_values, input_dict)

    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="recommendation-box">
            <strong>{i}.</strong> {rec}
        </div>
        """, unsafe_allow_html=True)

    # SHAP Analysis
    st.header(" Feature Impact Analysis")
    st.write("This chart shows how each factor influences your retirement readiness score:")

    fig = create_shap_visualization(shap_values, predictor.features, input_dict)
    if fig:
        st.pyplot(fig)
        plt.close(fig)

    # Feedback section
    st.header(" Feedback")
    feedback_manager = FeedbackManager()

    with st.form("feedback_form"):
        st.write("Help us improve our predictions by providing feedback:")

        feedback = st.radio(
            "How accurate was this prediction?",
            ["Very Accurate", "Somewhat Accurate", "Not Accurate"],
            horizontal=True
        )

        user_score = None
        if feedback == "Not Accurate":
            user_score = st.slider(
                "What do you think your actual readiness score should be?",
                min_value=0.0, max_value=1.0, value=float(score_rounded), step=0.01
            )

        additional_comments = st.text_area("Additional Comments (Optional)")

        if st.form_submit_button("Submit Feedback", type="primary"):
            final_feedback = f"{feedback}. {additional_comments}".strip()
            success = feedback_manager.save_feedback(input_dict, score_rounded, user_score, final_feedback)

            if success:
                st.success("Thank you for your feedback! This helps us improve our model.")
                st.session_state["shown_prediction"] = False
                st.balloons()


if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>© 2025 Wan Nur Shafiqah | Malaysian Retirement Readiness Predictor</p>
    <p style="font-size: 0.8rem;">Powered by Machine Learning & SHAP Explainable AI</p>
</div>
""", unsafe_allow_html=True)
