# 💼 Employee Attrition Prediction System

> Predictive analysis of employee attrition using machine learning techniques for actionable HR insights.

🔗 **Live Demo**: [Click to Open Streamlit App](https://employee-attrition-69vr2kxea9pzhbaxeub2ks.streamlit.app/)

---

## 📌 Overview

This project predicts whether an employee is at risk of attrition using various supervised machine learning models. The final model (Logistic Regression) is deployed via a **Streamlit** web application, enabling real-time predictions based on user input.

---

## 🎯 Objectives

- ✅ Predict if an employee is likely to leave the organization.
- ✅ Deliver a high-performing, interpretable, and user-friendly web application.
- ✅ Apply a real-world ML pipeline in the HR analytics domain.

---

## 🛠️ Tech Stack

| Category            | Tools / Libraries                         |
|---------------------|-------------------------------------------|
| **Language**        | Python                                    |
| **Data Analysis**   | pandas, numpy                             |
| **ML Modeling**     | scikit-learn, XGBoost                     |
| **Visualization**   | matplotlib, seaborn                       |
| **Deployment**      | Streamlit                                 |
| **Model Saving**    | joblib                                    |
| **Version Control** | Git, GitHub                               |

---

## 📁 Project Structure

```bash
.
├── app.py                     # Streamlit Web App
├── code.ipynb                 # Jupyter Notebook (EDA + Modeling)
├── logistic_model.joblib      # Trained Logistic Regression model
├── scaler.joblib              # Fitted StandardScaler object
├── columns.joblib             # Required input columns
├── HR-Employee-Attrition.csv  # Dataset
├── README.md                  # Project documentation
```

---

## 🧠 Methodology

### 🔄 Data Preprocessing

- One-hot encoding for categorical variables  
- Feature scaling using `StandardScaler`  
- Class imbalance handled using **SMOTE**

### ⚙️ Model Development

- Trained models:
  - ✅ Logistic Regression (Selected for deployment)
  - SVM
  - Random Forest
  - XGBoost
  - Decision Tree

- Logistic Regression was chosen due to:
  - High Accuracy (**90%**)
  - Balanced metrics and interpretability

### 📊 Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (Area Under Curve)
- SHAP values for feature importance explanation

### 🚀 Deployment

- Built an interactive web interface using **Streamlit**
- Model, scaler, and column metadata saved with `joblib`
- Input fields match training-time preprocessing

---

## 📈 Model Performance Summary

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| 0.90     | 0.83      | 0.43   | 0.57     | 0.87    |
| SVM                | 0.89     | 0.79      | 0.40   | 0.53     | 0.84    |
| Random Forest      | 0.86     | 0.88      | 0.12   | 0.21     | 0.76    |
| XGBoost            | 0.86     | 0.67      | 0.28   | 0.39     | 0.77    |
| Decision Tree      | 0.78     | 0.31      | 0.34   | 0.33     | 0.60    |

---

## 💡 Key Insights

- 🔥 **Overtime** is the strongest predictor of attrition.
- 💼 Other key features: **Job Satisfaction**, **Monthly Income**, **Work-Life Balance**
- ✅ Logistic Regression had the best balance of performance and interpretability.

---

## 📊 Visualizations

- ✅ Confusion Matrices
- ✅ ROC Curves
- ✅ SHAP Summary & Dependence Plots

---

## 🧩 Challenges Faced

- **Class Imbalance**: Solved using SMOTE, but overfitting had to be carefully avoided.
- **Feature Interpretability**: Not all features had business relevance and needed contextual analysis.
- **Pipeline Consistency**: Ensured deployment pipeline matches training logic.

---

## 🚀 Future Improvements

- Integrate **employee sentiment analysis** from surveys
- Offer **personalized HR recommendations**
- Expand explainability with **SHAP / LIME**
- Build **HR dashboard** for better decision insights

---

## 🧠 Learnings

- Developed full **end-to-end ML deployment** skills
- Gained insight into **HR analytics**
- Learned to **communicate technical results** in HR-friendly language
- Acknowledged importance of **ethics and fairness** in AI-based workplace systems

---

## 📚 Dataset

- **Source**: IBM HR Analytics Employee Attrition Dataset  
- **Records**: 1,470  
- **Features**: 35+ (e.g., `JobRole`, `OverTime`, `MonthlyIncome`, `JobSatisfaction`)

---

## 👩‍💻 Author

**Sonali Sinha**  
B.Tech CSE (AI & ML), Lovely Professional University  
📧 sonalisinha0610@gmail.com  
🔗 [GitHub](https://github.com/SONALISINHA01)

---

## 📄 License

**MIT License** – Feel free to fork, improve, and reuse with credits.

---

## 🌟 Star this repo if you found it helpful!
