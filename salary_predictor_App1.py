import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load the trained model (assumes you've saved it as a pickle file)
# knn_model = pickle.load(open('knn_model.pkl', 'rb'))

# Or, recreate a simple model here (you can modify based on your actual one)
# For now, let's define input ranges and do prediction inside app

st.set_page_config(page_title="UCI Income Classifier", layout="centered")
st.title("🔍 Salary Classification – UCI Adult Dataset")
st.markdown("Predict whether an individual's salary is **>50K** or **<=50K** based on demographic details.")

# --- Sidebar Input ---
st.sidebar.header("Enter User Details")

age = st.sidebar.slider("Age", 18, 90, 30)
education_num = st.sidebar.slider("Education Level (Numeric)", 1, 16, 10)
capital_gain = st.sidebar.number_input("Capital Gain", value=0)
capital_loss = st.sidebar.number_input("Capital Loss", value=0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)

# Dropdowns for categorical inputs (encoded manually)
workclass = st.sidebar.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Other"])
marital_status = st.sidebar.selectbox("Marital Status", ["Never-married", "Married-civ-spouse", "Divorced", "Other"])
occupation = st.sidebar.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Exec-managerial", "Other"])
relationship = st.sidebar.selectbox("Relationship", ["Not-in-family", "Husband", "Wife", "Own-child", "Other"])
race = st.sidebar.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Other"])
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
native_country = st.sidebar.selectbox("Native Country", ["United-States", "India", "Philippines", "Other"])

# Manual encoding
def encode_inputs():
    return [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        {"Private": 0, "Self-emp-not-inc": 1, "Local-gov": 2, "State-gov": 3, "Other": 4}[workclass],
        {"Never-married": 0, "Married-civ-spouse": 1, "Divorced": 2, "Other": 3}[marital_status],
        {"Tech-support": 0, "Craft-repair": 1, "Other-service": 2, "Exec-managerial": 3, "Other": 4}[occupation],
        {"Not-in-family": 0, "Husband": 1, "Wife": 2, "Own-child": 3, "Other": 4}[relationship],
        {"White": 0, "Black": 1, "Asian-Pac-Islander": 2, "Other": 3}[race],
        {"Male": 0, "Female": 1}[sex],
        {"United-States": 0, "India": 1, "Philippines": 2, "Other": 3}[native_country]
    ]

input_data = np.array([encode_inputs()])

# StandardScaler – optional (if used during training)
# scaler = pickle.load(open('scaler.pkl', 'rb'))
# input_data_scaled = scaler.transform(input_data)

# Simulate prediction using a threshold (for demo)
# Replace this with: prediction = knn_model.predict(input_data_scaled)
prediction = int((education_num > 9) and (hours_per_week > 40))

# --- Output ---
st.markdown("### 🔎 Prediction Result:")
if st.button("Predict Salary Class"):
    if prediction == 1:
        st.success("✅ Predicted Salary Class: >50K")
    else:
        st.warning("💼 Predicted Salary Class: <=50K")

st.markdown("---")
st.caption("Model based on UCI Adult Income Dataset – Streamlit Demo App by Devika 💡")
