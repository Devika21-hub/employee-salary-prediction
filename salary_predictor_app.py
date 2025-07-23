import streamlit as st
import pandas as pd
import joblib

# ✅ Load trained model and encoders
lr = joblib.load("model_lr.pkl")
le_edu = joblib.load("le_edu.pkl")
le_country = joblib.load("le_country.pkl")
model_columns = joblib.load("model_features.pkl")

# ✅ Page setup
st.set_page_config(page_title="Employee Salary Prediction", layout="centered")
st.title("💼 Employee Salary Prediction ")
st.write("Fill in your profile to estimate your annual salary based on the Stack Overflow Developer Survey 2024.")
# ✅ Currency map
currency_map = {
    "United States of America": ("USD", 1),
    "India": ("INR", 83),
    "Germany": ("EUR", 0.91),
    "Canada": ("CAD", 1.37),
    "United Kingdom of Great Britain and Northern Ireland": ("GBP", 0.78),
    "Australia": ("AUD", 1.48),
    "Brazil": ("BRL", 5.45),
    "Nigeria": ("NGN", 1540),
    "France": ("EUR", 0.91),
    "Japan": ("JPY", 157),
    "South Africa": ("ZAR", 18.4),
    "Mexico": ("MXN", 17.2),
    "Turkey": ("TRY", 33.5),
    "China": ("CNY", 7.3)
}

# ✅ User Inputs
education = st.selectbox("🎓 Education Level", le_edu.classes_)
experience = st.slider("🧠 Years of Experience", 0, 50, 3)
country = st.selectbox("🌍 Country", le_country.classes_)

remote_options = [
    "Remote",
    "In-person",
    "Hybrid (some remote, some in-person)",
    "Other"
]
remote_selected = st.selectbox("🏠 Work Environment", remote_options)

certified = st.radio("📜 Do you have any certifications?", ["Yes", "No"])

st.markdown("👨‍💻 **Select Your Developer Roles**:")
dev_types = [
    "Developer, back-end", "Developer, front-end", "Developer, full-stack",
    "Developer, mobile", "Developer, desktop or enterprise apps",
    "Data scientist or machine learning specialist", "Engineer, data",
    "Engineer, site reliability", "Academic researcher", "Student"
]
dev_selected = {role: st.checkbox(role) for role in dev_types}

st.markdown("💻 **Select Languages You Work With**:")
top_languages = [
    'JavaScript', 'HTML/CSS', 'SQL', 'Python', 'TypeScript',
    'Bash/Shell (all shells)', 'Java', 'C#', 'C++', 'C'
]
lang_selected = {lang: st.checkbox(lang) for lang in top_languages}

# ✅ Predict button
if st.button("🔍 Predict Salary"):
    try:
        # Encode categorical values
        edu_encoded = le_edu.transform([education])[0]
        country_encoded = le_country.transform([country])[0]
        cert_binary = 1 if certified == "Yes" else 0
        remote_dummies = [1 if remote_selected == opt else 0 for opt in remote_options]
        dev_binary = [1 if dev_selected[role] else 0 for role in dev_types]
        lang_binary = [1 if lang_selected[lang] else 0 for lang in top_languages]

        input_data = [edu_encoded, experience, country_encoded, cert_binary]
        input_data += remote_dummies + dev_binary + lang_binary

        # Create input DataFrame
        feature_names = (
            ["EdLevel", "YearsCodePro", "Country", "HasCertification"] +
            [f"Remote_{opt}" for opt in remote_options] +
            dev_types + top_languages
        )
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Align with model features
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_columns]

        # ✅ Predict and enforce minimum value
        raw_salary = lr.predict(input_df)[0]
        salary_usd = max(5000, raw_salary)  # prevent too-low predictions
        currency_code, rate = currency_map.get(country, ("USD", 1))
        salary_local = salary_usd * rate

        # ✅ Display result
        st.success("🎯 Estimated Salary Prediction")
        st.markdown(f"💵 **USD**: ${salary_usd:,.2f}")
        st.markdown(f"🌍 **{currency_code}**: {currency_code} {salary_local:,.2f}")

        # Professional explanation
        st.markdown("---")
        st.markdown(f"""
        ### 📌 Why this salary?
        This estimate is based on:
        - **Education Level:** `{education}`
        - **Experience:** `{experience} years`
        - **Country:** `{country}`
        - **Remote Work Preference:** `{remote_selected}`
        - **Certifications:** `{certified}`
        - **Developer Roles:** `{', '.join([role for role, selected in dev_selected.items() if selected]) or 'None'}`
        - **Languages Used:** `{', '.join([lang for lang, selected in lang_selected.items() if selected]) or 'None'}`
        """)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
