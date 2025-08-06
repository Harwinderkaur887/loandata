import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1605902711622-cfb43c4437b2");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Load the trained model
model = joblib.load('model.pkl')  # make sure this is the correct filename
scaler=joblib.load('scaler.pkl')
# Title
st.title("Loan Prediction App")

# Sidebar inputs
st.sidebar.header("Applicant Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.selectbox("Loan Term (months)", [360, 180, 120, 60])
credit_history = st.sidebar.selectbox("Credit History", [1.0, 0.0])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Convert to DataFrame for prediction
input_data = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area]
})

# Preprocessing should match the training preprocessing
# You must apply the same label encoding / one-hot encoding etc. here
# Example:
def preprocess(df):
    df = df.copy()
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
    df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)
    return df

input_processed = preprocess(input_data)

# Predict button
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_processed)[0]
    if prediction == 'Y':
        st.success("✅ Loan will be Approved!")
    else:


        st.error("❌ Loan will be Rejected.")

import matplotlib.pyplot as plt
import seaborn as sns

# Visual 1: Bar chart - Income vs Loan Amount
st.subheader("📊 Income vs Loan Amount")
fig, ax = plt.subplots()
bars = ax.bar(['Applicant Income', 'Coapplicant Income', 'Loan Amount'],
              [applicant_income, coapplicant_income, loan_amount],
              color=['skyblue', 'orange', 'green'])
ax.set_ylabel("Amount")
ax.set_title("Income and Loan Overview")
st.pyplot(fig)

# Visual 2: Pie chart - Property Area
st.subheader("🏘️ Property Area Distribution (Sample Input)")
area_counts = input_data['Property_Area'].value_counts()
fig2, ax2 = plt.subplots()
ax2.pie(area_counts, labels=area_counts.index, autopct='%1.1f%%', startangle=90)
ax2.set_title("Property Area Chosen")
st.pyplot(fig2)

# Visual 3: Credit History Gauge (simple)
st.subheader("💳 Credit History Status")
if credit_history == 1.0:
    st.success("Good Credit History")
else:
    st.warning("Poor or No Credit History")

# Visual 3: Credit History Status
    st.subheader("💳 Credit History Status")
    if credit_history == 1.0:
        st.success("Good credit history.")
    else:
        st.warning("Poor or no credit history.")

   










