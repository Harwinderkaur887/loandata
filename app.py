import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 Loan Eligibility Prediction App")
st.write("Enter applicant details below to check loan eligibility.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0, value=360)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Encode and scale input
def preprocess_input():
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    dependents_val = {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents]
    education_val = 1 if education == "Graduate" else 0
    self_employed_val = 1 if self_employed == "Yes" else 0
    property_area_val = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

    numerical = np.array([[applicant_income, coapplicant_income, loan_amount, loan_amount_term]])
    scaled_numerical = scaler.transform(numerical)

    return np.hstack(([
        gender_val,
        married_val,
        dependents_val,
        education_val,
        self_employed_val,
        credit_history,
        property_area_val
    ], scaled_numerical.flatten())).reshape(1, -1)

# Predict button
if st.button("Predict Loan Status"):
    try:
        input_data = preprocess_input()
        prediction = model.predict(input_data)

        if prediction[0] == "Y":
            st.success("✅ Loan will likely be Approved!")
        else:
            st.error("❌ Loan will likely be Rejected.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")








