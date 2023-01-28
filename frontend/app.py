"""This is the main entrypoint for the Streamlit app.

This module contains the Streamlit app and the API call to the backend.
"""

import random
import requests
import streamlit as st


st.title("Churn Prediction App")
st.write("This app predicts whether a customer will churn or not.")


gender = st.selectbox("Select your sex", ["Female", "Male"])

col1, col2, col3 = st.columns(3)

with col1:
    senior_citizen = st.radio("Are you a senior citizen?", ["Yes", "No"])

with col2:
    partner = st.radio("Do you have a partner?", ["Yes", "No"])

with col3:
    dependents = st.radio("Do you have dependents?", ["Yes", "No"])

tenure = st.slider("How many months have you been a customer?", 0, 72, 12)

col1, col2 = st.columns(2)

with col1:
    phone_service = st.radio("Do you have phone service?", ["Yes", "No"])

with col2:
    multiple_lines = st.radio("Do you have multiple lines?", ["Yes", "No"])

internet_service = st.selectbox(
    "Which internet service do you have?", ["DSL", "Fiber optic", "No"]
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    online_security = st.radio("Do you have online security?", ["Yes", "No"])

with col2:
    online_backup = st.radio("Do you have online backup?", ["Yes", "No"])

with col3:
    device_protection = st.radio(
        "Do you have device protection?", ["Yes", "No"]
    )

with col4:
    tech_support = st.radio("Do you have tech support?", ["Yes", "No"])

col1, col2, col3 = st.columns(3)

with col1:
    streaming_tv = st.radio("Do you have streaming TV?", ["Yes", "No"])

with col2:
    streaming_movies = st.radio("Do you have streaming movies?", ["Yes", "No"])

with col3:
    paperless_billing = st.radio(
        "Do you have paperless billing?", ["Yes", "No"]
    )

col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox(
        "Which contract do you have?",
        ["Month-to-month", "One year", "Two year"],
    )

with col2:
    payment_method = st.selectbox(
        "Which payment method do you use?",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ],
    )

col1, col2 = st.columns(2)

with col1:
    monthly_charges = st.slider(
        "How much do you pay per month?", 18.25, 118.75, 18.25
    )

with col2:
    total_charges = st.slider("How much have you paid in total?", 0, 5000, 500)


if st.button("Predict"):
    # Create a random customerID with the same length as the original data
    # with numbers between 1000 and 9999
    customer_id = (
        str(random.randint(1000, 9999)) + "-" + str(random.randint(1000, 9999))
    )

    data = {
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": 72,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=data)

    prediction = response.json()["churn_prediction"]
    probability = response.json()["churn_probability"]

    if prediction == 1:
        st.write(
            "This customer will churn. with a probability of", probability
        )
    else:
        st.write(
            "This customer will not churn. with a probability of", probability
        )
