import streamlit as st
import pandas as pd
import joblib




st.set_page_config(
    page_title="AI Financial Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# Load models
fraud_model = joblib.load("fraud_detection_model.pkl")
loan_model = joblib.load("loan_default_model.pkl")
risk_model = joblib.load("risk_scoring_model.pkl")
anomaly_model = joblib.load("transaction_anomaly_model.pkl")
cluster_model = joblib.load("spending_cluster_model.pkl")

# Sidebar
st.sidebar.title("AI Fraud Detection System")

page = st.sidebar.selectbox(
    "Select Module",
    [
        "Home",
        "Credit Card Fraud Detection",
        "Loan Default Prediction",
        "Customer Risk Scoring",
        "Transaction Anomaly Detection",
        "Spending Pattern Clustering"
    ]
)

# ---------------- HOME PAGE ----------------

if page == "Home":

    st.title("AI Powered Financial Fraud Detection System")

    st.image("images/bank.png", width=600)

    st.markdown("""
    ### Project Modules

    - Credit Card Fraud Detection  
    - Loan Default Prediction  
    - Customer Risk Scoring  
    - Transaction Anomaly Detection  
    - Spending Pattern Clustering
    """)

# ---------------- FRAUD DETECTION ----------------

elif page == "Credit Card Fraud Detection":

    st.header("Credit Card Fraud Detection")
    
    time = st.number_input("Transaction Time")
    amount = st.number_input("Transaction Amount")

    if st.button("Detect Fraud"):
    
        features = [[time, amount]]
    
        pred = fraud_model.predict(features)
    
        if pred[0] == 1:
            st.error("Fraud Detected")
        else:
            st.success("Normal Transaction")

# ---------------- LOAN DEFAULT ----------------

elif page == "Loan Default Prediction":

    st.header("Loan Default Prediction")

    age = st.number_input("Age")
    income = st.number_input("Income")
    loan = st.number_input("Loan Amount")

    if st.button("Predict Loan Default"):

        pred = loan_model.predict([[age, income, loan]])

        if pred[0] == 1:
            st.error("Customer Likely to Default")
        else:
            st.success("Customer Safe")

# ---------------- RISK SCORING ----------------

elif page == "Customer Risk Scoring":

    st.header("Customer Risk Scoring")

    income = st.number_input("Income")
    credit = st.number_input("Credit Score")

    if st.button("Calculate Risk"):

        prob = risk_model.predict_proba([[income, credit]])[0][1]

        score = prob * 100

        st.metric("Risk Score", round(score,2))

# ---------------- ANOMALY DETECTION ----------------

elif page == "Transaction Anomaly Detection":

    st.header("Transaction Anomaly Detection")

    amount = st.number_input("Transaction Amount")

    if st.button("Check Transaction"):

        pred = anomaly_model.predict([[amount]])

        if pred[0] == -1:
            st.error("Anomalous Transaction")
        else:
            st.success("Normal Transaction")

# ---------------- CLUSTERING ----------------

elif page == "Spending Pattern Clustering":

    st.header("Spending Pattern Clustering")

    spend = st.number_input("Customer Monthly Spending")

    if st.button("Find Cluster"):

        cluster = cluster_model.predict([[spend]])

        st.write("Customer Cluster:", cluster[0])