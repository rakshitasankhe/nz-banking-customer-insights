import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="NZ Banking Churn Predictor", layout="centered")

st.title("🏦 NZ Banking Customer Churn Predictor")

# Load model and training data features once
@st.cache_data
def load_model():
    model = joblib.load('output/random_forest_model.pkl')
    return model

@st.cache_data
def load_features():
    # Assuming you saved feature names or load from your training data CSV headers
    return ['Age', 'Tenure', 'AccountBalance', 'IsActive', 'NumOfProducts', 'CreditScore']

model = load_model()
features = load_features()

# Input widgets
st.sidebar.header("Input Customer Data")

age = st.sidebar.slider("Age", 18, 90, 35)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 3)
account_balance = st.sidebar.number_input("Account Balance ($)", min_value=0.0, max_value=100000.0, value=5000.0, step=100.0)
is_active = st.sidebar.selectbox("Is Account Active?", options=[1, 0], format_func=lambda x: "Yes" if x==1 else "No")
num_products = st.sidebar.slider("Number of Products", 1, 5, 1)
