# Gender -> 1 Female 0 Male
# Churn -> 1 Yes  0 No
# Scaler is exported as scaler.pkl
# Model is exported as best_model.pkl
# Order of the X ->'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st
import joblib   
import numpy as np

scaler = joblib.load("scaler.pkl")  
model = joblib.load("best_model.pkl")   


st.title("Churn Prediction App")

st.divider()

st.write("Please enter the following details to predict if a customer will churn or not:")  

st.divider()

age = st.number_input("Age", min_value=18, max_value=100, value=30)


tenure = st.number_input("Tenure (in months)", min_value=0, max_value=130, value=10) 

monthly_charges = st.number_input("Monthly Charges", min_value=30.0,max_value=150.0, value=50.0)

gender = st.selectbox("Gender", options=["Male", "Female"]) 

st.divider()

predict_button = st.button("Predict Churn")

if predict_button:
    gender_value = 1 if gender  == "Female" else 0      

    X = np.array([[age, gender_value, tenure, monthly_charges]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    probability = model.predict_proba(X_scaled)
    if prediction[0] == 1:
        st.error(f"The customer is likely to churn with a probability of {probability[0][1]:.2f}")
    else:
        st.success(f"The customer is not likely to churn with a probability of {probability[0][0]:.2f}")        

else:
    st.info("Please fill in the details and click the Predict Churn button to see the result.")
