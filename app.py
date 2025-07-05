import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="NZ Banking Churn Predictor", layout="centered")
st.title("🏦 NZ Banking Customer Churn Predictor")

@st.cache_data
def load_model():
    return joblib.load('output/random_forest_model.pkl')

model = load_model()

# Sidebar inputs
st.sidebar.header("Input Customer Data")
age = st.sidebar.slider("Age", 18, 90, 35)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 3)
account_balance = st.sidebar.number_input("Account Balance ($)", 0.0, 100000.0, 5000.0, step=100.0)
is_active = st.sidebar.selectbox("Is Account Active?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
num_products = st.sidebar.slider("Number of Products", 1, 5, 1)
credit_score = st.sidebar.slider("Credit Score", 300, 900, 600)

# Prepare input dataframe
input_df = pd.DataFrame({
    'Age': [age],
    'Tenure': [tenure],
    'AccountBalance': [account_balance],
    'IsActive': [is_active],
    'NumOfProducts': [num_products],
    'CreditScore': [credit_score]
})

# Predict churn probability
probability = model.predict_proba(input_df)[:, 1][0]
prediction = "❌ Will Churn" if probability > 0.5 else "✅ Will Stay"

# Display results
st.subheader("Prediction Results")
st.write(f"Churn Probability: **{probability:.2f}**")
st.write(f"Prediction: **{prediction}**")
