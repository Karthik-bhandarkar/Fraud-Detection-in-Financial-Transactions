import streamlit as st
import pandas as pd
import joblib

# --------------------------
# Load the trained model pipeline
# --------------------------
model = joblib.load('fraud_detection_pipeline.pickle')

# --------------------------
# Feature Engineering Function
# --------------------------
def feature_engineering(data):
    """Applies feature engineering steps to the input data."""
    data['balanceDiffOrig'] = data['oldbalanceOrg'] - data['newbalanceOrig']
    data['balanceDiffDest'] = data['newbalanceDest'] - data['oldbalanceDest']
    return data

# --------------------------
# Streamlit App UI
# --------------------------
st.title("💳 Fraud Detection Prediction App")
st.markdown("Enter transaction details manually OR upload a CSV file for bulk predictions.")
st.divider()

# --------------------------
# Manual Input Section
# --------------------------
st.subheader("🔹 Single Transaction Input")

transaction_type = st.selectbox(
    "Transaction Type",
    ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
)

amount = st.number_input("Amount", min_value=0.0, value=1000.0, format="%f")
old_balance_original = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0, format="%f")
new_balance_original = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0, format="%f")
old_balance_destination = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0, format="%f")
new_balance_destination = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0, format="%f")

if st.button("Predict Single Transaction"):
    input_data = pd.DataFrame([{
        'type': transaction_type,
        'amount': amount,
        'oldbalanceOrg': old_balance_original,
        'newbalanceOrig': new_balance_original,
        'oldbalanceDest': old_balance_destination,
        'newbalanceDest': new_balance_destination
    }])

    # Apply feature engineering
    input_data_fe = feature_engineering(input_data.copy())

    prediction = model.predict(input_data_fe)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("🚨 This transaction is likely FRAUD!")
    else:
        st.success("✅ This transaction looks safe (Not Fraud).")

# --------------------------
# CSV Upload Section
# --------------------------
st.divider()
st.subheader("📂 Bulk Prediction via CSV Upload")

uploaded_file = st.file_uploader("Upload a CSV file with transactions", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Transactions Preview")
    st.dataframe(input_data.head())

    # Apply feature engineering
    input_data_fe = feature_engineering(input_data.copy())

    # Predictions
    predictions = model.predict(input_data_fe)

    # Add results
    input_data["Prediction"] = predictions
    input_data["Prediction_Label"] = input_data["Prediction"].map({0: "Not Fraud", 1: "Fraud"})

    st.subheader("Predictions")
    st.dataframe(input_data)

    # Download button
    csv_out = input_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv_out,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )
