import streamlit as st
import requests

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Medical Insurance Predictor",
    page_icon="🏥",
    layout="wide"
)

# ---------------- BACKGROUND + STYLING ---------------- #
st.markdown("""
<style>

/* Background Image with Soft Overlay */
[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(rgba(255,255,255,0.65), rgba(255,255,255,0.65)),
        url("https://images.unsplash.com/photo-1576091160550-2173dba999ef");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Main Title */
.main-title {
    font-size: 40px;
    font-weight: 700;
    color: #0a3d62;
    text-align: center;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #444;
    margin-bottom: 30px;
}

/* Form Card */
.form-card {
    background-color: rgba(255,255,255,0.95);
    padding: 35px;
    border-radius: 18px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
}

/* Prediction Box */
.prediction-box {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(to right, #0a3d62, #3c6382);
    color: white;
    text-align: center;
    font-size: 26px;
    font-weight: 600;
    margin-top: 25px;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 50px;
    color: gray;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
st.markdown('<div class="main-title">🏥 Medical Insurance Cost Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Based Healthcare Expense Estimation System</div>', unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000/predict"

# ---------------- INPUT FORM ---------------- #
with st.container():
    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🩺 Personal Details")
        age = st.number_input("Age", 18, 100)
        sex = st.selectbox("Sex", ["Male", "Female"])
        bmi = st.number_input("BMI", 10.0, 50.0)
        smoker = st.selectbox("Smoker", ["Yes", "No"])

    with col2:
        st.subheader("💳 Insurance Details")
        income = st.number_input("Annual Income (₹)")
        region = st.selectbox("Region", ["North", "South", "East", "West"])
        claims_count = st.number_input("Claims Count", 0, 50)
        annual_premium = st.number_input("Annual Premium (₹)")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICT BUTTON ---------------- #
if st.button("🔍 Predict Medical Cost", use_container_width=True):

    input_data = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "smoker": smoker,
        "income": income,
        "region": region,
        "claims_count": claims_count,
        "annual_premium": annual_premium
    }

    try:
        with st.spinner("Analyzing medical profile..."):
            response = requests.post(API_URL, json=input_data)

        if response.status_code == 200:
            result = response.json()
            predicted_cost = result["predicted_annual_medical_cost"]

            st.markdown(
                f'<div class="prediction-box">💰 Predicted Annual Medical Cost: ₹ {predicted_cost:,.2f}</div>',
                unsafe_allow_html=True
            )
        else:
            st.error("⚠ Unable to connect to prediction API.")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# ---------------- FOOTER ---------------- #
st.markdown(
    '<div class="footer">Healthcare Analytics System • Built with Streamlit & FastAPI</div>',
    unsafe_allow_html=True
)