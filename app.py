%%writefile app.py

import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ABC Ltd - Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

bundle = joblib.load("churn_model.pkl")

clf = bundle["clf_model"]
encoders = bundle["encoders"]
features_clf = bundle["features_clf"]

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("ABC Ltd")
st.subheader("Customer Churn Prediction Tool")

st.write(
    """
    This tool uses a predictive model to estimate the probability
    that a customer may churn. It is intended to support managerial
    decision-making and should be used alongside managerial judgement.
    """
)

st.divider()

# --------------------------------------------------
# Customer inputs
# --------------------------------------------------

st.header("Customer Information")

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=840.0
)

# Get categories from the encoders
gender = st.selectbox(
    "Gender",
    encoders["gender"].classes_.tolist()
)

partner = st.selectbox(
    "Partner",
    encoders["Partner"].classes_.tolist()
)

dependents = st.selectbox(
    "Dependents",
    encoders["Dependents"].classes_.tolist()
)

phone_service = st.selectbox(
    "Phone Service",
    encoders["PhoneService"].classes_.tolist()
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    encoders["MultipleLines"].classes_.tolist()
)

internet_service = st.selectbox(
    "Internet Service",
    encoders["InternetService"].classes_.tolist()
)

online_security = st.selectbox(
    "Online Security",
    encoders["OnlineSecurity"].classes_.tolist()
)

online_backup = st.selectbox(
    "Online Backup",
    encoders["OnlineBackup"].classes_.tolist()
)

device_protection = st.selectbox(
    "Device Protection",
    encoders["DeviceProtection"].classes_.tolist()
)

tech_support = st.selectbox(
    "Tech Support",
    encoders["TechSupport"].classes_.tolist()
)

streaming_tv = st.selectbox(
    "Streaming TV",
    encoders["StreamingTV"].classes_.tolist()
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    encoders["StreamingMovies"].classes_.tolist()
)

contract = st.selectbox(
    "Contract",
    encoders["Contract"].classes_.tolist()
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    encoders["PaperlessBilling"].classes_.tolist()
)

payment_method = st.selectbox(
    "Payment Method",
    encoders["PaymentMethod"].classes_.tolist()
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    encoders["SeniorCitizen"].classes_.tolist()
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button("🔮 Predict Churn Risk", use_container_width=True):

    # Create customer record
    customer = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # Encode categorical variables exactly as training data
    for col, encoder in encoders.items():

        if col in customer.columns:
            customer[col] = encoder.transform(customer[col])

    # Ensure exact feature order
    customer = customer[features_clf]

    # Predict probability
    probability = clf.predict_proba(customer)[0][1]

    percentage = probability * 100

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        "Predicted Churn Probability",
        f"{percentage:.1f}%"
    )

    # Risk classification
    if probability >= 0.70:

        st.error("🔴 HIGH CHURN RISK")

        st.write(
            """
            The model estimates a relatively high probability of churn.
            Management may consider a targeted retention intervention.
            """
        )

    elif probability >= 0.40:

        st.warning("🟡 MEDIUM CHURN RISK")

        st.write(
            """
            The customer shows a moderate predicted risk of churn.
            Management may consider monitoring the customer and reviewing
            possible retention actions.
            """
        )

    else:

        st.success("🟢 LOW CHURN RISK")

        st.write(
            """
            The model estimates a relatively low probability of churn.
            No immediate intervention may be required based on this prediction.
            """
        )

    st.info(
        "Important: This prediction is a decision-support output and "
        "should be considered alongside managerial judgement and other "
        "customer information."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "ABC Ltd | Predictive Analytics & Managerial AI Adoption Assignment"
)
