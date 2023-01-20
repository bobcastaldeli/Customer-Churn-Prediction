"""This is the main entrypoint for the Streamlit app.

This module contains the Streamlit app and the API call to the backend.
"""


import requests
import streamlit as st


st.title("Churn Prediction App")

st.write("This app predicts whether a customer will churn or not.")


col1, col2, col3, col4 = st.columns(4)

with col1:
    gender = st.radio("Select your gender", ["Female", "Male"])

with col2:
    senior_citizen = st.radio("Are you a senior citizen?", ["Yes", "No"])

with col3:
    partner = st.radio("Do you have a partner?", ["Yes", "No"])

with col4:
    dependents = st.radio("Do you have dependents?", ["Yes", "No"])

tenure = st.slider("How many months have you been a customer?", 0, 72, 12)

col1, col2 = st.columns(2)

with col1:
    phone_service = st.radio("Do you have phone service?", ["Yes", "No"])

with col2:
    multiple_lines = st.radio("Do you have multiple lines?", ["Yes", "No"])

col1, col2, col3, col4 = st.columns(4)

with col1:
    internet_service = st.radio(
        "Which internet service do you have?", ["DSL", "Fiber optic", "No"]
    )

with col2:
    online_security = st.radio("Do you have online security?", ["Yes", "No"])

with col3:
    online_backup = st.radio("Do you have online backup?", ["Yes", "No"])

with col4:
    device_protection = st.radio(
        "Do you have device protection?", ["Yes", "No"]
    )

col1, col2, col3, col4 = st.columns(4)

with col1:
    tech_support = st.radio("Do you have tech support?", ["Yes", "No"])

with col2:
    streaming_tv = st.radio("Do you have streaming TV?", ["Yes", "No"])

with col3:
    streaming_movies = st.radio("Do you have streaming movies?", ["Yes", "No"])

with col4:
    contract = st.radio(
        "Which contract do you have?",
        ["Month-to-month", "One year", "Two year"],
    )

col1, col2 = st.columns(2)

with col1:
    paperless_billing = st.radio(
        "Do you have paperless billing?", ["Yes", "No"]
    )

with col2:
    payment_method = st.radio(
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
