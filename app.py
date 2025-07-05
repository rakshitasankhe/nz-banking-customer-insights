import streamlit as st
import pandas as pd
import joblib

# Load the trained Random Forest model
model = joblib.load('output/random_forest_model.pkl')

st.title("🏦 NZ Banking Churn Predictor")

# Input widgets matching your model's features
age = st.slider("Age", 18, 90, 30)
tenure = st.slider("Tenure (Years)", 0, 10, 3)
account_balance = st.number_input("Account Balance", min_value=0, value=5000)
credit_score = st.slider("Credit Score", 300, 900, 650)
is_active = st.selectbox("Is Active (1=Yes, 0=No)", options=[1, 0], index=0)
num_of_products = st.selectbox("Number of Products", options=[1, 2, 3, 4], index=0)

# Prepare input DataFrame with the exact feature order
input_df = pd.DataFrame({
    'Age': [age],
    'Tenure': [tenure],
    'AccountBalance': [account_balance],
    'IsActive': [is_active],
    'NumOfProducts': [num_of_products],
    'CreditScore': [credit_score]
})

if st.button("Predict Churn"):
    probability = model.predict_proba(input_df)[:, 1][0]
    prediction = model.predict(input_df)[0]

    st.write(f"### Churn Probability: {probability:.2f}")
    if prediction == 1:
        st.markdown("### ❌ Likely to Churn")
    else:
        st.markdown("### ✅ Likely to Stay")
