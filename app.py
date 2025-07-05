import streamlit as st
import pandas as pd
import joblib

model = joblib.load("output/random_forest_model.pkl")

st.title("🏦 NZ Banking Churn Predictor")

age = st.slider("Age", 18, 90, 30)
balance = st.number_input("Account Balance")
tenure = st.slider("Tenure (Years)", 0, 10)
credit_score = st.slider("Credit Score", 300, 900)

# Create input
input_df = pd.DataFrame([[age, tenure, balance, 1, 2, credit_score]],
                        columns=['Age', 'Tenure', 'AccountBalance', 'IsActive', 'NumOfProducts', 'CreditScore'])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    st.success(f"Churn Probability: {proba:.2f}")
    st.write("Prediction:", "⚠️ Will Churn" if prediction == 1 else "✅ Will Stay")
