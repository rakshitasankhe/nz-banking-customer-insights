# NZ Banking Customer Churn Analysis - Summary

## Project Overview
This project analyzes customer data from a New Zealand bank to predict customer churn using machine learning models. The goal is to identify customers at risk of leaving and understand key factors driving churn.

## Model Performance
- **Random Forest Classifier**  
  - Accuracy: 0.87  
  - ROC AUC Score: 0.91  
- **Logistic Regression**  
  - Accuracy: 0.82  
  - ROC AUC Score: 0.85  

The Random Forest model performed best with higher accuracy and ROC AUC, indicating effective churn prediction.

## Key Feature Insights
- **Account Balance:** Customers with lower balances tend to churn more.  
- **Tenure:** Longer tenure reduces churn likelihood.  
- **Age:** Younger customers show slightly higher churn rates.

## Recommendations
- Focus retention efforts on customers with low balances and short tenure.  
- Consider personalized offers or engagement programs targeting younger customers.

## Limitations and Next Steps
- Additional features like customer feedback or transaction history could improve models.  
- Explore advanced models (XGBoost, SVM) and hyperparameter tuning for better accuracy.  
- Deploy model as a web app for real-time churn prediction.

---


