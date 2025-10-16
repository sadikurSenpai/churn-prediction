import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessor
model = joblib.load("best_random_forest_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("Customer Churn Prediction App")
st.markdown("Enter customer details below to predict the churn probability.")

# ---- User Inputs ----
age = st.number_input("Age", min_value=18, max_value=100, value=30)
annual_income = st.number_input("Annual Income (k$)", min_value=0.0, value=50.0)
total_spend = st.number_input("Total Spend", min_value=0.0, value=1000.0)
num_of_purchases = st.number_input("Number of Purchases", min_value=0, value=10)
avg_transaction = st.number_input("Average Transaction Amount", min_value=0.0, value=100.0)
num_support_contacts = st.number_input("Number of Support Contacts", min_value=0, value=1)
satisfaction_score = st.slider("Satisfaction Score (1–5)", 1, 5, 3)
last_purchase_days_ago = st.number_input("Days Since Last Purchase", min_value=0, value=100)
email_opt_in = st.selectbox("Email Opt-In", ["TRUE", "FALSE"])

# Convert email opt-in to boolean
email_opt_in_bool = True if email_opt_in == "TRUE" else False

# Create DataFrame for model
input_data = pd.DataFrame([{
    "Age": age,
    "Annual_Income": annual_income,
    "Total_Spend": total_spend,
    "Num_of_Purchases": num_of_purchases,
    "Average_Transaction_Amount": avg_transaction,
    "Num_of_Support_Contacts": num_support_contacts,
    "Satisfaction_Score": satisfaction_score,
    "Last_Purchase_Days_Ago": last_purchase_days_ago,
    "Email_Opt_In": email_opt_in_bool
}])

st.write("### Input Summary")
st.dataframe(input_data)

# ---- Prediction ----
if st.button("Predict Churn"):
    # Preprocess input
    processed_input = preprocessor.transform(input_data)
    
    # Predict
    churn_prob = model.predict_proba(processed_input)[0][1]
    churn_pred = model.predict(processed_input)[0]
    
    st.markdown("---")
    st.subheader(" Prediction Result:")
    st.write(f"**Predicted Churn:** {'YES' if churn_pred == 1 else 'NO'}")
    st.write(f"**Churn Probability:** {churn_prob:.2f}")

    if churn_pred == 1:
        st.error(" The customer is likely to churn.")
    else:
        st.success(" The customer is likely to stay.")

