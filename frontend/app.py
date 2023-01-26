"""This is the main entrypoint for the Streamlit app.

This module contains the Streamlit app and the API call to the backend.
"""


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
    data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if phone_service == "Yes" else 0,
        "MultipleLines": 1 if multiple_lines == "Yes" else 0,
        "InternetService": internet_service,
        "OnlineSecurity": 1 if online_security == "Yes" else 0,
        "OnlineBackup": 1 if online_backup == "Yes" else 0,
        "DeviceProtection": 1 if device_protection == "Yes" else 0,
        "TechSupport": 1 if tech_support == "Yes" else 0,
        "StreamingTV": 1 if streaming_tv == "Yes" else 0,
        "StreamingMovies": 1 if streaming_movies == "Yes" else 0,
        "Contract": contract,
        "PaperlessBilling": 1 if paperless_billing == "Yes" else 0,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    st.write(data)

    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    # prediction = response.json()["churn_prediction"]
    # probability = response.json()["churn_probability"]

    st.write(f"Prediction: {response}")
    # st.write(f"Probability: {probability}")
