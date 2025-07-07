
import streamlit as st
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Sample training to simulate a trained model and encoder
train_data = pd.DataFrame({
    'Age_y': [45, 50, 65],
    'Gender_y': ['Male', 'Female', 'Male'],
    'Blood Group': ['A+', 'B+', 'A-'],
    'Required Organ': ['Kidney', 'Heart', 'Kidney'],
    'Location': ['Dhaka', 'Chittagong', 'Dhaka'],
    'Condition_y': ['Critical', 'Stable', 'Critical'],
    'Age_x': [43, 45, 54],
    'Gender_x': ['Female', 'Male', 'Male'],
    'Donated Organ': ['Heart', 'Kidney', 'Kidney'],
    'Condition_x': ['Healthy', 'Healthy', 'Critical']
})
train_labels = ['No Matched', 'Matched', 'No Matched']
X_train = train_data.drop('Condition_y', axis=1)
y_train = train_labels

# Column transformer and model
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'),
                   ['Age_y', 'Gender_y', 'Blood Group', 'Required Organ', 'Location',
                    'Age_x', 'Gender_x', 'Donated Organ', 'Condition_x'])],
    remainder='passthrough'
)
X_encoded = ct.fit_transform(X_train)
model = RandomForestClassifier()
model.fit(X_encoded, y_train)

# Streamlit App
st.title("🧬 Organ Transplantation Eligibility Prediction")

st.header("🔍 Enter Patient Information")
patient_age = st.number_input("Patient's Age", min_value=0, max_value=100)
patient_gender = st.selectbox("Patient's Gender", ["Male", "Female"])
patient_blood = st.selectbox("Patient's Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
patient_organ = st.selectbox("Required Organ", ["Kidney", "Heart", "Liver", "Lung", "Pancreas"])
patient_location = st.selectbox("Patient's Location", ["Dhaka", "Chittagong", "Sylhet", "Rajshahi", "Barishal"])
patient_condition = st.selectbox("Patient's Condition", ["Critical", "Stable"])

st.header("🧑‍⚕️ Enter Donor Information")
donor_age = st.number_input("Donor's Age", min_value=0, max_value=100)
donor_gender = st.selectbox("Donor's Gender", ["Male", "Female"])
donor_blood = st.selectbox("Donor's Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
donor_organ = st.selectbox("Donated Organ", ["Kidney", "Heart", "Liver", "Lung", "Pancreas"])
donor_location = st.selectbox("Donor's Location", ["Dhaka", "Chittagong", "Sylhet", "Rajshahi", "Barishal"])
donor_condition = st.selectbox("Donor's Condition", ["Healthy", "Critical"])

if st.button("🔮 Predict Eligibility"):
    match_score = int(patient_blood == donor_blood and patient_organ == donor_organ and patient_location == donor_location)
    condition_factor = 0.5 if patient_condition == "Critical" and donor_condition == "Critical" else 1
    posibility = match_score * condition_factor

    input_data = pd.DataFrame({
        'Age_y': [patient_age],
        'Gender_y': [patient_gender],
        'Blood Group': [patient_blood],
        'Required Organ': [patient_organ],
        'Location': [patient_location],
        'Condition_y': [patient_condition],
        'Age_x': [donor_age],
        'Gender_x': [donor_gender],
        'Donated Organ': [donor_organ],
        'Location': [donor_location],
        'Condition_x': [donor_condition]
    })

    input_encoded = ct.transform(input_data)
    prediction = model.predict(input_encoded)[0]

    st.subheader("🧾 Result")
    st.write(f"🧪 Match Score: `{match_score}`")
    st.write(f"📊 Eligibility Score: `{posibility}`")
    st.write(f"📌 Prediction: `{prediction}`")
    if posibility >= 0.95:
        st.success("✅ Congratulations! You are eligible for the transplantation process.")
    else:
        st.warning("❌ You are not eligible for transplantation at this time.")
