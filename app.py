import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="NZ Banking Churn Predictor", layout="centered")
st.title("🏦 NZ Banking Customer Churn Predictor")

# Load model and features once
@st.cache_data
def load_model():
    return joblib.load('output/random_forest_model.pkl')

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
credit_score = st.sidebar.slider("Credit Score", 300, 850, 600)

# Dataset upload for visualizations
uploaded_file = st.sidebar.file_uploader("Upload cleaned_nz_banking_data.csv", type=["csv"])

if uploaded_file:
    @st.cache_data
    def load_data(uploaded_file):
        return pd.read_csv(uploaded_file)

    df = load_data(uploaded_file)
    st.sidebar.success("Data loaded successfully!")
else:
    df = None
    st.sidebar.warning("Upload dataset for visualizations")

# Predict churn on button click
if st.sidebar.button("Calculate Churn Probability"):
    input_df = pd.DataFrame({
        'Age': [age],
        'Tenure': [tenure],
        'AccountBalance': [account_balance],
        'IsActive': [is_active],
        'NumOfProducts': [num_products],
        'CreditScore': [credit_score]
    })

    pred_proba = model.predict_proba(input_df)[:, 1][0]
    pred_class = model.predict(input_df)[0]

    st.subheader("Prediction Results")
    st.write(f"**Churn Probability:** {pred_proba:.2%}")
    st.write(f"**Predicted Churn Class:** {'Yes' if pred_class == 1 else 'No'}")

# Data visualizations
if df is not None:
    st.header("📊 Data Visualizations")

    # Churn distribution
    churn_counts = df['Churn'].value_counts()
    st.bar_chart(churn_counts)

    # Age distribution by churn
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x='Age', hue='Churn', multiple='stack', bins=20)
    st.pyplot(plt.gcf())
    plt.clf()

    # Feature importance
    feat_imp = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
    st.pyplot(plt.gcf())
    plt.clf()

# Dummy model comparison - Replace these with your actual trained model metrics
st.header("📈 Model Comparison")
metrics_df = pd.DataFrame({
    'Model': ['Random Forest', 'Logistic Regression'],
    'Accuracy': [0.81, 0.47],
    'ROC AUC': [0.50, 0.47]
})

fig, ax = plt.subplots(1, 2, figsize=(1
