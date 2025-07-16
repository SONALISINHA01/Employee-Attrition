import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Load model and artifacts
model = joblib.load("logistic_model.joblib")
scaler = joblib.load("scaler.joblib")
required_columns = joblib.load("columns.joblib")

# SHAP explainer setup (using scaled background)
@st.cache_resource
def load_explainer():
    background = scaler.transform(np.zeros((1, len(required_columns))))
    return shap.LinearExplainer(model, background)

explainer = load_explainer()

# UI
st.title("🧠 Employee Attrition Predictor")
st.write("Fill the following employee information to predict if they are likely to leave.")

# Sidebar inputs
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

# Preprocessing
def preprocess_input(data):
    data = pd.get_dummies(data)
    for col in required_columns:
        if col not in data.columns:
            data[col] = 0
    return data[required_columns]

# SHAP helper
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Prediction
if st.button("Predict Attrition"):
    processed_input = preprocess_input(input_data)
    scaled_input = scaler.transform(processed_input)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This employee is likely to leave. (Probability: {probability:.2f})")
    else:
        st.success(f"✅ This employee is likely to stay. (Probability: {probability:.2f})")

    # SHAP explanation using scaled input
    shap_values = explainer.shap_values(scaled_input)

    st.subheader("🔍 Why this prediction?")
    
    # Waterfall plot (legacy for LinearExplainer)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, 
        shap_values[0], 
        feature_names=processed_input.columns,
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig)


    # SHAP feature importance
    shap_df = pd.DataFrame({
        'Feature': processed_input.columns,
        'SHAP Value': shap_values[0]
    }).sort_values(by="SHAP Value", key=abs, ascending=False)

    st.write("📌 Top features influencing this prediction:")
    st.dataframe(shap_df.head(5))
