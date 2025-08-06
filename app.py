import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.markdown(
    """
    <style>
    /* Make sidebar background black */
    [data-testid="stSidebar"] {
        background-color: #000000;
        color: white;
    }

    /* Optional: Change input text and label colors */
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] .stTextInput,
    [data-testid="stSidebar"] .stNumberInput,
    [data-testid="stSidebar"] .stSelectbox {
        color: white;
    }

    /* Optional: change header color in sidebar */
    .sidebar .sidebar-content h1,
    .sidebar .sidebar-content h2,
    .sidebar .sidebar-content h3 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/free-photo/top-view-finances-elements-arrangement-with-copy-space_23-2148793719.jpg?ga=GA1.1.806108855.1754501801&semt=ais_incoming&w=740&q=80"); /* You can replace this with any direct image link */
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

   # --- Generate downloadable PDF report ---

# Predict button
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_processed)[0]
    
    if prediction == 'Y':
        st.success("✅ Loan will be Approved!")
    else:
        st.error("❌ Loan will be Rejected.")

    # --- Generate downloadable PDF report ---
    result_text = "✅ Loan Approved!" if prediction == 'Y' else "❌ Loan Rejected."
    
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Loan Prediction Result")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Prediction: {result_text}")
    c.drawString(50, height - 130, f"Applicant Income: {applicant_income}")
    c.drawString(50, height - 150, f"Coapplicant Income: {coapplicant_income}")
    c.drawString(50, height - 170, f"Loan Amount: {loan_amount}")
    c.drawString(50, height - 190, f"Loan Term: {loan_term} months")
    c.drawString(50, height - 210, f"Credit History: {'Good' if credit_history == 1.0 else 'Poor/None'}")
    c.drawString(50, height - 230, f"Property Area: {property_area}")

    c.drawString(50, height - 270, "Thank you for using the Loan Prediction App!")
    
    c.showPage()
    c.save()
    
    buffer.seek(0)
    
    st.download_button(
        label="📥 Download PDF",
        data=buffer,
        file_name="loan_prediction_result.pdf",
        mime="application/pdf"
    )
















