import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
model = joblib.load('fraud_detection_pipeline.pickle')

# App title and markdown
st.title("Fraud Detection Prediction App")
st.markdown("Please enter the transaction details and use the predict button.")
st.divider()

# Input fields
transaction_type = st.selectbox(
    "Transaction Type",
    ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
)

amount = st.number_input("Amount", min_value=0.0, value=1000.0, format="%f")

old_balance_original = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0, format="%f")

new_balance_original = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0, format="%f")

old_balance_destination = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0, format="%f")

new_balance_destination = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0, format="%f")

# Prediction button
if st.button("Predict"):
    # Create input DataFrame (Note: The app code shown does not create the 'balance_difference' features
    # that the model was trained on, which may cause a prediction error.)
    input_data = pd.DataFrame([{
        'type': transaction_type,
        'amount': amount,
        'oldbalanceOrg': old_balance_original,
        'newbalanceOrig': new_balance_original,
        'oldbalanceDest': old_balance_destination,
        'newbalanceDest': new_balance_destination
    }])

    prediction = model.predict(input_data)[0]

    st.subheader(f"Prediction: {int(prediction)}")

    if prediction == 1:
        st.error("This transaction can be fraud!")
    else:
        st.success("This transaction looks like it is not a fraud.")