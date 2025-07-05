import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="NZ Banking Churn Predictor", layout="centered")

st.title("🏦 NZ Banking Customer Churn Predictor")

# Load model and feature list once
@st.cache_data
def load_model():
    model = joblib.load('output/random_forest_model.pkl')
    return model

@st.cache_data
def load_features():
    return ['Age', 'Tenure', 'AccountBalance', 'IsActive', 'NumOfProducts', 'CreditScore']

model = load_model()
features = load_features()

# Sidebar inputs
st.sidebar.header("Input Customer Data")

age = st.sidebar.slider("Age", 18, 90, 35)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 3)
account_balance = st.sidebar.number_input("Account Balance ($)", min_value=0.0, max_value=100000.0, value=5000.0, step=100.0)
is_active = st.sidebar.selectbox("Is Account Active?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
num_products = st.sidebar.slider("Number of Products", 1, 5, 1)
credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)

# Button to trigger prediction
if st.sidebar.button("Calculate Churn Probability"):

    # Prepare input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'Tenure': [tenure],
        'AccountBalance': [account_balance],
        'IsActive': [is_active],
        'NumOfProducts': [num_products],
        'CreditScore': [credit_score]
    })

    # Predict churn probability
    churn_proba = model.predict_proba(input_data)[:, 1][0]

    st.subheader("Prediction Results")
    st.write(f"Churn Probability: {churn_proba:.2f}")

    if churn_proba > 0.5:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")

# Optional: Show model comparison metrics
st.header("📊 Model Comparison")

metrics_df = pd.DataFrame({
    'Model': ['Random Forest', 'Logistic Regression'],
    'Accuracy': [0.81, 0.47],
    'ROC AUC': [0.50, 0.47]
})

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

sns.barplot(x='Model', y='Accuracy', data=metrics_df, ax=ax[0], palette='Blues')
ax[0].set_ylim(0, 1)
ax[0].set_title('Accuracy Comparison')

sns.barplot(x='Model', y='ROC AUC', data=metrics_df, ax=ax[1], palette='Greens')
ax[1].set_ylim(0, 1)
ax[1].set_title('ROC AUC Comparison')

st.pyplot(fig)
