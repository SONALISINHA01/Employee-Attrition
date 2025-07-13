import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and artifacts
model = joblib.load("logistic_model.joblib")
scaler = joblib.load("scaler.joblib")
required_columns = joblib.load("columns.joblib")

# UI
st.title("🧠 Employee Attrition Predictor")
st.write("Fill the following employee information to predict if they are likely to leave.")

# Sidebar input fields
age = st.sidebar.slider("Age", 18, 60, 30)
distance = st.sidebar.slider("Distance From Home (km)", 1, 30, 5)
monthly_income = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
hourly_rate = st.sidebar.slider("Hourly Rate", 10, 100, 50)
daily_rate = st.sidebar.slider("Daily Rate", 100, 1500, 800)
monthly_rate = st.sidebar.slider("Monthly Rate", 1000, 30000, 15000)
job_level = st.sidebar.selectbox("Job Level", [1, 2, 3, 4, 5])
education = st.sidebar.selectbox("Education Level", [1, 2, 3, 4, 5])
education_field = st.sidebar.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"])
department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
business_travel = st.sidebar.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
job_role = st.sidebar.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 3)
work_life_balance = st.sidebar.slider("Work Life Balance (1-4)", 1, 4, 3)
environment_satisfaction = st.sidebar.slider("Environment Satisfaction (1-4)", 1, 4, 3)
relationship_satisfaction = st.sidebar.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
job_involvement = st.sidebar.slider("Job Involvement (1-4)", 1, 4, 3)
num_companies_worked = st.sidebar.slider("Number of Companies Worked", 0, 10, 2)
percent_salary_hike = st.sidebar.slider("Percent Salary Hike", 0, 30, 10)
performance_rating = st.sidebar.selectbox("Performance Rating", [1, 2, 3, 4])
training_times_last_year = st.sidebar.slider("Training Times Last Year", 0, 10, 3)
total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 10)
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)
years_in_current_role = st.sidebar.slider("Years in Current Role", 0, 20, 3)
years_since_last_promotion = st.sidebar.slider("Years Since Last Promotion", 0, 15, 1)
years_with_curr_manager = st.sidebar.slider("Years With Current Manager", 0, 20, 2)

# Create input dataframe
input_data = pd.DataFrame({
    'Age': [age],
    'DistanceFromHome': [distance],
    'MonthlyIncome': [monthly_income],
    'HourlyRate': [hourly_rate],
    'DailyRate': [daily_rate],
    'MonthlyRate': [monthly_rate],
    'JobLevel': [job_level],
    'Education': [education],
    'JobSatisfaction': [job_satisfaction],
    'WorkLifeBalance': [work_life_balance],
    'EnvironmentSatisfaction': [environment_satisfaction],
    'RelationshipSatisfaction': [relationship_satisfaction],
    'JobInvolvement': [job_involvement],
    'NumCompaniesWorked': [num_companies_worked],
    'PercentSalaryHike': [percent_salary_hike],
    'PerformanceRating': [performance_rating],
    'TrainingTimesLastYear': [training_times_last_year],
    'TotalWorkingYears': [total_working_years],
    'YearsAtCompany': [years_at_company],
    'YearsInCurrentRole': [years_in_current_role],
    'YearsSinceLastPromotion': [years_since_last_promotion],
    'YearsWithCurrManager': [years_with_curr_manager],
    'BusinessTravel': [business_travel],
    'Department': [department],
    'EducationField': [education_field],
    'Gender': [gender],
    'JobRole': [job_role],
    'MaritalStatus': [marital_status],
    'OverTime': [overtime]
})

# Preprocess input like training
def preprocess_input(data):
    data = pd.get_dummies(data)

    for col in required_columns:
        if col not in data.columns:
            data[col] = 0

    data = data[required_columns]
    data_scaled = scaler.transform(data)
    return data_scaled

# Predict
if st.button("Predict Attrition"):
    processed = preprocess_input(input_data)
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This employee is likely to leave. (Probability: {probability:.2f})")
    else:
        st.success(f"✅ This employee is likely to stay. (Probability: {probability:.2f})")
