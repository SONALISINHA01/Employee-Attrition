💼 Employee Attrition Prediction System
Predictive analysis of employee attrition using machine learning techniques for actionable HR insights.

📌 Project Overview
This project aims to predict employee attrition using supervised learning algorithms, with a focus on Logistic Regression, supported by SVM, Random Forest, XGBoost, and Decision Trees. The system was deployed using Streamlit, allowing real-time predictions based on user input.



https://employee-attrition-69vr2kxea9pzhbaxeub2ks.streamlit.app/


🎯 Objectives
Predict if an employee is likely to leave the organization.

Deliver a high-performing, interpretable, and user-friendly web application.

Build HR analytics skills using real-world ML pipelines.

📊 Technologies & Tools Used
Category	Tools/Technologies
Language	Python
Data Analysis	pandas, numpy
ML Modeling	scikit-learn, XGBoost
Visualization	matplotlib, seaborn
Model Deployment	Streamlit
Model Serialization	joblib
Version Control	Git & GitHub

📁 Project Structure
bash
Copy
Edit
.
├── app.py                     # Streamlit Web App
├── code.ipynb                # Jupyter Notebook with training, evaluation
├── logistic_model.joblib     # Trained logistic regression model
├── scaler.joblib             # Fitted StandardScaler object
├── columns.joblib            # Required input columns used during training
├── HR-Employee-Attrition.csv # Dataset used for training
├── README.md                 # Project documentation
🧠 Methodology
Data Preprocessing

One-hot encoding of categorical variables

Feature scaling using StandardScaler

Addressed class imbalance using SMOTE

Model Development

Models trained: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost

Logistic Regression chosen for its accuracy (90%) and interpretability

Evaluation

Accuracy, Precision, Recall, F1-Score

ROC Curve & AUC scores

SHAP values used for feature explainability

Deployment

Built an interactive web interface using Streamlit

Input fields dynamically match model’s trained features

📈 Model Performance Summary
Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	0.90	0.83	0.43	0.57	0.87
SVM	0.89	0.79	0.40	0.53	0.84
Random Forest	0.86	0.88	0.12	0.21	0.76
XGBoost	0.86	0.67	0.28	0.39	0.77
Decision Tree	0.78	0.31	0.34	0.33	0.60

💡 Key Insights
Overtime is a strong predictor of attrition.

Job Satisfaction, Monthly Income, and Work-Life Balance are key influencers.

Logistic Regression provided the best tradeoff between accuracy and interpretability.

🔍 Visualizations
Confusion Matrices

ROC Curves

SHAP plots for feature importance

🧩 Challenges Faced
Class Imbalance: Managed using SMOTE but required careful evaluation to prevent overfitting.

Feature Relevance: Not all features were business-explainable; required domain reasoning.

Deployment Complexity: Ensuring preprocessing logic matches between training and inference.

🚀 Future Improvements
Integrate employee survey sentiment data

Add personalized HR recommendations per prediction

Expand SHAP/LIME visualizations for transparency

Build dashboard for HR teams with analytics insights

🧠 Learnings
Built end-to-end ML deployment experience

Improved understanding of HR metrics and their predictive relevance

Learned to communicate ML results in business-friendly language

Realized the importance of ethics and bias mitigation in HR AI models

📚 Dataset
Source: IBM HR Analytics Employee Attrition & Performance dataset

Records: 1,470

Features: 14+ (e.g., JobRole, Overtime, MonthlyIncome)

👩‍💻 Author
Sonali Sinha
B.Tech CSE (AI & ML), Lovely Professional University
📧 sonalisinha0610@gmail.com

