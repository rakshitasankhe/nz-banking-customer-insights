import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import seaborn as sns

# Load saved model and data (adjust path if needed)
@st.cache_data
def load_model():
    model = joblib.load('output/random_forest_model.pkl')
    return model

@st.cache_data
def load_data():
    df = pd.read_csv('output/cleaned_nz_banking_data.csv')
    return df

model = load_model()
df = load_data()

st.title("🏦 NZ Banking Churn Predictor")

# Input sliders with ranges based on your dataset
age = st.slider("Age", int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].median()))
balance = st.slider("Account Balance", int(df['AccountBalance'].min()), int(df['AccountBalance'].max()), int(df['AccountBalance'].median()))
tenure = st.slider("Tenure (Years)", int(df['Tenure'].min()), int(df['Tenure'].max()), int(df['Tenure'].median()))
credit_score = st.slider("Credit Score", int(df['CreditScore'].min()), int(df['CreditScore'].max()), int(df['CreditScore'].median()))

# Prepare input for prediction
input_df = pd.DataFrame({
    'Age': [age],
    'AccountBalance': [balance],
    'Tenure': [tenure],
    'CreditScore': [credit_score],
    # Add other required features here, with default or typical values if needed
})

# Predict churn probability
probability = model.predict_proba(input_df)[:, 1][0]
prediction = "✅ Will Stay" if probability < 0.5 else "❌ Likely to Churn"

st.markdown(f"### Churn Probability: {probability:.2%}")
st.markdown(f"### Prediction: {prediction}")

# Load test set (adjust if you saved separately)
# For demo, use same df for evaluation - replace with your test set in practice
X = df[['Age', 'AccountBalance', 'Tenure', 'CreditScore']]  # adjust features accordingly
y = df['Churn']

# Split train/test in your notebook, here just showing evaluation on full dataset
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:,1]

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
roc_auc = roc_auc_score(y, y_proba)

st.markdown("## Model Performance Metrics")
st.write(f"Accuracy: {accuracy:.3f}")
st.write(f"Precision: {precision:.3f}")
st.write(f"Recall: {recall:.3f}")
st.write(f"ROC AUC: {roc_auc:.3f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y, y_proba)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax.plot([0,1], [0,1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
st.pyplot(fig)

# Feature importance
importances = model.feature_importances_
features = X.columns

feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

fig2, ax2 = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=feat_imp_df, ax=ax2, palette='viridis')
ax2.set_title("Feature Importances")
st.pyplot(fig2)

