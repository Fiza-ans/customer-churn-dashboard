import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# ---------------- LOAD MODEL ----------------
model = joblib.load("final_model.pkl")

# ---------------- TITLE ----------------
st.title("📉 Customer Churn Prediction App")
st.subheader("🔍 Real-time Churn Prediction using Machine Learning")
st.caption("Model: XGBoost | Features: Customer profile + RFM + Service bundle")

# ---------------- PERFORMANCE SECTION ----------------
st.header("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC Score", "0.835")
col2.metric("F1 Score", "0.573")
col3.metric("Precision", "0.635")
col4.metric("Recall", "0.521")

# ---------------- FEATURE IMPORTANCE ----------------
st.header("⭐ Feature Importance")

try:
    # If model is pipeline
    if hasattr(model, "named_steps"):
        importances = model.named_steps['model'].feature_importances_
    else:
        importances = model.feature_importances_

    fig, ax = plt.subplots()
    ax.barh(range(len(importances[:10])), importances[:10])
    ax.set_title("Top Features Influencing Churn")

    st.pyplot(fig)

except Exception as e:
    st.info("Feature importance not available for this model.")

st.write("Fill the details to predict customer churn")

# ---------------- INPUTS ----------------
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner?", ["Yes", "No"])
Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=100.0)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

# ---------------- ENGINEERED FEATURES ----------------
tenure_ratio = tenure / 72 if tenure > 0 else 0

Frequency = st.number_input("Frequency", min_value=0, value=1)
Monetary = st.number_input("Monetary", min_value=0.0, value=100.0)
Recency = st.number_input("Recency", min_value=0.0, value=0.5)

service_bundle_score = st.slider("Service Bundle Score", 0, 5, 2)
payment_reliability_score = st.slider("Payment Reliability Score", 0, 5, 2)

# ---------------- INPUT DATAFRAME ----------------
input_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "tenure_ratio": tenure_ratio,
    "Frequency": Frequency,
    "Monetary": Monetary,
    "Recency": Recency,
    "service_bundle_score": service_bundle_score,
    "payment_reliability_score": payment_reliability_score
}])

# ---------------- PREDICTION ----------------
prob = None

if st.button("Predict Churn"):

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn (Probability: {prob:.2f})")
        st.caption("This customer shows patterns similar to users who previously churned.")
    else:
        st.success(f"✅ Low Risk of Churn (Probability: {prob:.2f})")
        st.caption("This customer shows stable behavior and lower churn risk.")

    if prob >= 0.7:
        st.warning("🔴 Risk Level: Very High")
    elif prob >= 0.5:
        st.warning("🟠 Risk Level: Medium")
    else:
        st.info("🟢 Risk Level: Low")
    # -------- CHURN RISK GAUGE --------
    st.subheader("🎯 Churn Risk Score")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Churn Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"},
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

# ---------------- RESET ----------------
st.divider()

if st.button("Reset Form"):
    st.rerun()

# ---------------- RECOMMENDATIONS ----------------
if prob is not None:

    st.header("💡 Retention Recommendations")

    if prob >= 0.7:
        st.write("• Offer discount or loyalty reward")
        st.write("• Provide premium customer support")
        st.write("• Recommend long-term contract plans")

    elif prob >= 0.5:
        st.write("• Send engagement emails")
        st.write("• Offer bundled services")
        st.write("• Monitor customer activity")

    else:
        st.write("• Maintain engagement")
        st.write("• Offer upsell opportunities")