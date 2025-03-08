import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("salary_model.pkl")
encoder = joblib.load("salary_encoder.pkl")

st.title("Salary-Based Expense Prediction")
st.write("Enter your details below to predict your estimated monthly expense.")

salary = st.number_input("Enter your salary", min_value=30, max_value=2000000, step=1)
level = st.number_input("Enter your level (1-10)", min_value=1, max_value=10, step=1)
qualification = st.selectbox("Select your qualification",
                             ["noformal", "secondary", "certificate", "nce", "bsc", "msc", "phd", "prof"])

new_data = pd.DataFrame({
    "salary": [salary],
    "level": [level],
    "qualification": [qualification.lower()]
})

prediction_data = encoder.transform(new_data)

if st.button("Predict Expense"):
    predicted_expense = model.predict(prediction_data)[0]

    st.success(f"### Predicted Expense: N{predicted_expense:.2f}")

    # Show entered details
    st.write(f"**Salary:** N{salary}")
    st.write(f"**Level:** {level}")
    st.write(f"**Qualification:** {qualification}")

