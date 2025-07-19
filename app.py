import streamlit as st
import pickle
import pandas as pd

# Load the model
with open("salary_model.pkl", "rb") as f:
    model = pickle.load(f)

# Manual encoders
gender_map = {"Male": 0, "Female": 1, "Other": 2}
edu_map = {
    "High School": 0,
    "Bachelor's": 1,
    "Master's": 2,
    "PhD": 3
}
job_map = {
    "Software Engineer": 0,
    "Data Scientist": 1,
    "HR Manager": 2,
    "Accountant": 3,
    "Marketing Executive": 4
}

# Page config
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

# Custom styles and logo
st.markdown("""
    <style>
        body {
            background-color: #dff3fd;
        }
        .main-title {
            font-size: 40px;
            color: #2c3e50;
            text-align: center;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .salary-box {
            font-size: 24px;
            border: 2px solid #d32f2f;
            padding: 18px;
            border-radius: 12px;
            background-color: #fff5f0;
            text-align: center;
            color: #b71c1c;
            font-weight: bold;
            margin-top: 25px;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            color: #888;
        }
        .logo {
            text-align: center;
            margin-bottom: 10px;
        }
    </style>

    <div class="logo">
        <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" width="90"/>
    </div>
    <div class="main-title">👩‍💼💼 <u>Employee Salary Predictor</u> 💰📊</div>
""", unsafe_allow_html=True)

st.write("### 📝 Enter your details below:")

# Centered form layout
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    age = st.number_input("📅 Age", min_value=18, max_value=65, value=25, help="Enter your current age.")
    gender = st.selectbox("⚧️ Gender", list(gender_map.keys()), help="Choose your gender.")
    education = st.selectbox("🎓 Education Level", list(edu_map.keys()), help="Select your highest qualification.")
    job_title = st.selectbox("💼 Job Title", list(job_map.keys()), help="Choose your profession/job title.")
    experience = st.slider("🧪 Years of Experience", 0, 40, 1, help="Years of work experience.")

# Predict button
if st.button("🚀 Predict Salary"):
    try:
        # Encode inputs
        encoded_input = pd.DataFrame([[
            age,
            gender_map[gender],
            edu_map[education],
            job_map[job_title],
            experience
        ]], columns=['age', 'gender', 'education level', 'job title', 'years of experience'])

        predicted_salary = model.predict(encoded_input)[0]

        # Rule-based fallback for unrealistic predictions
        if predicted_salary < 5000 or predicted_salary > 100000:
            job_base_salary = {
                0: 18000,  # Software Engineer
                1: 20000,  # Data Scientist
                2: 15000,  # HR Manager
                3: 14000,  # Accountant
                4: 16000   # Marketing Executive
            }

            job_encoded = job_map[job_title]
            base_salary = job_base_salary.get(job_encoded, 15000)

            # Age factor
            if age < 23:
                age_factor = 0.9
            elif 23 <= age <= 27:
                age_factor = 1.0
            elif 28 <= age <= 35:
                age_factor = 1.1
            elif 36 <= age <= 45:
                age_factor = 1.2
            else:
                age_factor = 1.3

            # Experience factor
            if experience == 0:
                exp_factor = 0.9
            elif 1 <= experience <= 2:
                exp_factor = 1.0
            elif 3 <= experience <= 5:
                exp_factor = 1.2
            elif 6 <= experience <= 9:
                exp_factor = 1.4
            elif 10 <= experience <= 15:
                exp_factor = 1.6
            else:
                exp_factor = 1.8

            predicted_salary = int(base_salary * age_factor * exp_factor)

        # Final salary display (monthly and yearly)
        monthly_salary = int(predicted_salary)
        yearly_salary = monthly_salary * 12

        st.markdown(
            f'''
            <div class="salary-box">
                💰 <b>Estimated Monthly Salary</b>: ₹{monthly_salary:,}<br>
                🗓️ <b>Estimated Yearly Salary</b>: ₹{yearly_salary:,}
            </div>
            ''',
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

# Model of Evaluation section (replacing old footer)
st.markdown('<h4 style="text-align:center; margin-top:30px;">📊 Model of Evaluation</h4>', unsafe_allow_html=True)

# Show local images (JPG) with proper layout
st.image("model_eval1.png.jpg", caption="Model Evaluation - Chart 1", use_container_width=True)
st.image("model_eval2.png.jpg", caption="Model Evaluation - Chart 2", use_container_width=True)
