

# 🎓 Student Success Prediction Dashboard: Predicting Dropout, Enrollment, and Graduation

An end-to-end Machine Learning project that predicts higher-education student outcomes (**Dropout**, **Continue (Enrolled)**, or **Graduate**) based on a blend of academic, demographic, and economic factors.

This project delivers a **complete ML pipeline** from raw data ingestion and preprocessing to model training, evaluation, and an interactive application built with **Streamlit**.

-----

## ✨ Features & Project Overview

This project analyzes higher-education student data to identify patterns and key predictors of academic success, creating a powerful tool for institutional intervention.

### **Key Components**

| Component | Description | Technologies |
| :--- | :--- | :--- |
| **Data Pipeline** | Cleaning, encoding, feature engineering, and train/test split. | Pandas, NumPy |
| **Model Training** | Training and optimizing **Logistic Regression** and **Random Forest** models. | Scikit-learn |
| **Evaluation & Analysis** | Classification reports, confusion matrices, and in-depth **Feature Importance** analysis. | Scikit-learn, Matplotlib, Seaborn |
| **Interactive Dashboard** | A web application for data storytelling, live prediction, and **What-If Scenarios**. | Streamlit, Plotly |

### **Model Performance Summary**

| Model | Purpose | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Baseline Model | $0.70$ | $0.71$ |
| **Random Forest (Optimized)** | **Final Production Model** | **$0.74$** | **$0.74$** |

### **🎯 Key Insights**

  * **Academic performance (Average Grades)** and **Financial Conditions** are the primary predictors of student success.
  * Students with high average grades and no reported financial issues are significantly more likely to graduate.
  * **Economic indicators** (Inflation, GDP, Unemployment) show a secondary but visible impact on student outcomes.
  * The use of **SMOTE** (Synthetic Minority Over-sampling Technique) was critical for balancing the dataset, significantly improving the model's performance on the minority classes (Dropout and Enrolled).

-----

## 🛠️ Installation & Setup

### **1. Clone the Repository**

```bash
git clone https://github.com/MohamedGamal-Ahmed/student_success_project.git
cd student_success_project
```

### **2. Environment Setup**

Create and activate a dedicated virtual environment to manage dependencies:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment
# On Windows:
# .venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### **3. Install Dependencies**

Install all required libraries using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

-----

## 🚀 How to Run the Project

The project is executed in a sequential pipeline before launching the interactive dashboard.

### **Pipeline Execution Order**

Run the following scripts in order from the project root directory:

```bash
# 1. Preprocess and clean data
python src/data_preprocessing.py

# 2. Generate new academic and financial features
python src/feature_engineering.py

# 3. Train models and save the best one
python src/model_training.py

# 4. Evaluate trained models and generate report
python src/model_evaluation.py

# 5. Analyze and save feature importance
python src/feature_analysis.py
```

### **Running the Streamlit Dashboard**

After running the data pipeline, launch the interactive application:

```bash
streamlit run app/streamlit_app.py
```

Open the link shown in your terminal (usually: `Local URL: http://localhost:8501`).

### **🧭 App Pages Overview**

  * **📊 Data Story:** Visualizations of data distributions, correlations, and feature statistics.
  * **🔮 Quick Predictor:** Simple form for fast prediction using 9 key features (e.g., age, average grade, financial issues).
  * **📈 Advanced Analysis:** Deep dive into feature importance, correlations, and the impact of economic factors.
  * **🤔 What-If Scenarios:** Allows users to simulate changes (e.g., improving grades or removing financial issues) to see how the prediction changes in real-time.

-----

## 📂 Project Structure

```
student_success_project/
│
├── app/
│ └── streamlit_app.py  # Interactive dashboard app
│
├── data/
│ ├── raw_students.csv  # Original dataset
│ └── ...               # Cleaned and split datasets
│
├── models/
│ ├── model.pkl
│ └── random_forest_optimized.pkl  # Final, optimized model
│
├── reports/
│ ├── model_evaluation.txt
│ └── feature_importance.csv
│
├── src/
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ └── ...                 # All other pipeline scripts
│
└── requirements.txt
```

-----

## 🧑‍💻 Author

**Mohamed Gamal**

🎯 AI Engineer | Machine Learning Enthusiast

  * **Email**: mgamal.ahmed@outlook.com
  * **LinkedIn**: [https://www.linkedin.com/in/mohamed-gamal-357b10356](https://www.linkedin.com/in/mohamed-gamal-357b10356)
  * **GitHub**: [https://github.com/MohamedGamal-Ahmed](https://github.com/MohamedGamal-Ahmed)

