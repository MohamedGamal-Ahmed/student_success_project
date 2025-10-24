import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import plotly.express as px
from sklearn.feature_selection import SelectKBest, f_classif
import os

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="🎓 Student Success Prediction Dashboard",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Success Prediction Dashboard")
st.markdown("""
Welcome to the **Student Retention Project** — a full data journey from exploration and model building to prediction insights.
""")

# =========================
# Load Data & Models
# =========================
@st.cache_resource
def load_models_and_data():
    """Load all models and necessary data, caching the result."""
    try:
        # Load models
        model_simple_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'C:/Users/Mgama/Ai_Amit_Diploma/Amit_ai_diploma/student_success_project/models/model_optimized.pkl')
        model_advanced_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'C:/Users/Mgama/Ai_Amit_Diploma/Amit_ai_diploma/student_success_project/models/random_forest_optimized.pkl')
        
        models = {
            "Simple Model (9 features)": joblib.load(model_simple_path),
            "Advanced Model (34 features)": joblib.load(model_advanced_path)
        }

        # Load data for analysis and feature selection
        clean_data_path = os.path.join(os.path.dirname(__file__), '..', 'C:/Users/Mgama/Ai_Amit_Diploma/Amit_ai_diploma/student_success_project/data/clean_students.csv')
        x_train_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'C:/Users/Mgama/Ai_Amit_Diploma/Amit_ai_diploma/student_success_project/data/X_train.csv')
        y_train_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'C:/Users/Mgama/Ai_Amit_Diploma/Amit_ai_diploma/student_success_project/data/y_train.csv')
        
        data = {
            "full_data": pd.read_csv(clean_data_path),
            "X_train": pd.read_csv(x_train_path),
            "y_train": pd.read_csv(y_train_path).values.ravel()
        }
        return models, data
    except FileNotFoundError as e:
        st.error(f"❌ Error loading resources: {e}. Please ensure all model and data files are present.")
        return None, None

models, data = load_models_and_data()
TARGET_MAP = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
REVERSE_TARGET_MAP = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}

# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("📍 Dashboard Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Data Story", "🔮 Quick Predictor", "📈 Advanced Analysis", "🤔 What-If Scenarios"]
)

# ==============================================================================
# PAGE 1: DATA STORY
# ==============================================================================
if page == "📊 Data Story" and data:
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.write("Let's explore the dataset before modeling.")
    
    df = data["full_data"].copy()
    
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", df.shape[0])
    c2.metric("Total Features", df.shape[1] - 1)
    c3.metric("Unique Outcomes", df["Target"].nunique())
    target_counts = df["Target"].value_counts()
    c4.metric("Dropout Rate", f"{(target_counts.get(0, 0) / len(df) * 100):.1f}%")

    # Target Distribution
    st.subheader("🎯 Student Outcomes Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        target_mapped = df["Target"].map(TARGET_MAP).value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#E97121", "#2196F3", "#4CAF50"]
        ax.bar(target_mapped.index, target_mapped.values, color=colors)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Number of Students per Outcome Category", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(target_mapped.values):
            ax.text(i, v + 20, str(v), ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        target_pct = (target_mapped.values / target_mapped.sum()) * 100
        wedges, texts, autotexts = ax.pie(target_pct, labels=target_mapped.index, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        ax.set_title("Percentage Distribution", fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("📈 Correlation Heatmap (Top Features)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 1:
        # Select top 10 features for better visualization
        corr_matrix = df[numeric_cols].corr()
        # Get correlations with target
        target_corr = corr_matrix['Target'].abs().sort_values(ascending=False)[:10]
        selected_features = list(target_corr.index)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[selected_features].corr(), annot=True, fmt='.2f', cmap="coolwarm", 
                   ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation Matrix - Top 10 Features", fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    # Dataset Statistics
    st.subheader("📊 Dataset Statistics")
    st.dataframe(df.describe(), use_container_width=True)

# ==============================================================================
# PAGE 2: QUICK PREDICTOR (9 Features)
# ==============================================================================
if page == "🔮 Quick Predictor" and models:
    st.header("🔮 Quick Student Outcome Prediction")
    st.write("Fill in the student's basic information for a quick prediction.")
    st.info("💡 This model uses 9 key features for fast predictions.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age at enrollment", 17, 70, 20, help="Student's age when enrolling")
        avg_grade = st.number_input("Average Grade (0-20)", 0.0, 20.0, 12.0, step=0.1)
        avg_success_rate = st.number_input("Success Rate (0-1)", 0.0, 1.0, 0.8, step=0.01)
    
    with col2:
        financial_issue = st.selectbox("Financial Issue", [0, 1], 
                                       format_func=lambda x: "No" if x == 0 else "Yes")
        scholarship_holder = st.selectbox("Scholarship Holder", [0, 1],
                                         format_func=lambda x: "No" if x == 0 else "Yes")
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    
    with col3:
        attendance = st.selectbox("Attendance Type", [1, 0], 
                                 format_func=lambda x: "Daytime" if x == 1 else "Evening")
        units_1st = st.number_input("Units 1st Semester", 0, 30, 6)
        units_2nd = st.number_input("Units 2nd Semester", 0, 30, 6)

    if st.button("🔍 Make Prediction", use_container_width=True):
        try:
            # Get feature names from training data
            feature_names = data["X_train"].columns
            
            # Create input dataframe
            user_input_df = pd.DataFrame([[
                age, avg_grade, avg_success_rate, financial_issue, scholarship_holder, 
                gender, attendance, units_1st, units_2nd
            ]], columns=feature_names)

            # Process input for the simple model (feature selection)
            selector = SelectKBest(score_func=f_classif, k=8)
            selector.fit(data["X_train"], data["y_train"])
            processed_input = selector.transform(user_input_df)

            # Get prediction
            simple_model = models["Simple Model (9 features)"]
            prediction = simple_model.predict(processed_input)[0]
            probabilities = simple_model.predict_proba(processed_input)[0]
            
            # Display results
            st.divider()
            st.subheader("🎯 Prediction Result")
            
            predicted_label = TARGET_MAP.get(prediction, 'Unknown')
            confidence = max(probabilities) * 100
            
            col1, col2 = st.columns([2, 1])
            with col1:
                if prediction == 0:
                    st.error(f"⚠️ **Predicted Outcome:** {predicted_label}", icon="🚨")
                elif prediction == 1:
                    st.warning(f"📌 **Predicted Outcome:** {predicted_label}", icon="⏳")
                else:
                    st.success(f"✅ **Predicted Outcome:** {predicted_label}", icon="🎓")
                    st.balloons()
            
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")
            
            # Probability breakdown
            st.subheader("📊 Prediction Probabilities")
            prob_df = pd.DataFrame({
                "Outcome": [TARGET_MAP[i] for i in [0, 1, 2]],
                "Probability (%)": [prob * 100 for prob in probabilities]
            })
            
            fig = px.bar(prob_df, x="Outcome", y="Probability (%)", 
                        color="Outcome", title="Outcome Probabilities",
                        color_discrete_map={'Dropout': '#E97121', 'Enrolled': '#2196F3', 'Graduate': '#4CAF50'})
            fig.update_yaxes(range=[0, 100])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")

# ==============================================================================
# PAGE 3: ADVANCED ANALYSIS
# ==============================================================================
if page == "📈 Advanced Analysis" and data:
    st.header("📈 Advanced Data Analysis")
    df = data["full_data"].copy()
    df['Target_Label'] = df['Target'].map(TARGET_MAP)

    # 1. Age and Academic Success
    st.subheader("1️⃣ Relationship between Age and Academic Success")
    fig_age = px.histogram(df, x='Age at enrollment', color='Target_Label', 
                          barmode='overlay', marginal='box',
                          title="Distribution of Age by Academic Outcome",
                          color_discrete_map={'Dropout': '#E97121', 'Enrolled': '#2196F3', 'Graduate': '#4CAF50'})
    st.plotly_chart(fig_age, use_container_width=True)

    # 2. Economic Factors
    st.subheader("2️⃣ Impact of Economic Factors")
    eco_factors = [col for col in df.columns if 'Unemployment' in col or 'Inflation' in col or 'GDP' in col]
    
    if eco_factors:
        selected_factor = st.selectbox("Select an economic factor to analyze:", eco_factors)
        fig_eco = px.box(df, x='Target_Label', y=selected_factor, color='Target_Label',
                        title=f"Impact of {selected_factor} on Academic Outcome",
                        color_discrete_map={'Dropout': '#E97121', 'Enrolled': '#2196F3', 'Graduate': '#4CAF50'})
        st.plotly_chart(fig_eco, use_container_width=True)

    # 3. Feature Importance
    st.subheader("3️⃣ Feature Importance Analysis")
    selector = SelectKBest(score_func=f_classif, k=len(data["X_train"].columns))
    selector.fit(data["X_train"], data["y_train"])
    
    feature_importance = pd.DataFrame({
        'Feature': data["X_train"].columns,
        'Score': selector.scores_
    }).sort_values('Score', ascending=False).head(15)
    
    fig_imp = px.bar(feature_importance, x='Score', y='Feature', orientation='h',
                    title="Top 15 Most Important Features",
                    labels={'Score': 'Feature Score', 'Feature': 'Feature Name'})
    fig_imp.update_traces(marker_color='#2196F3')
    st.plotly_chart(fig_imp, use_container_width=True)

# ==============================================================================
# PAGE 4: WHAT-IF SCENARIOS
# ==============================================================================
if page == "🤔 What-If Scenarios" and models:
    st.header("🤔 What-If Scenarios")
    st.write("Interactively change student data to see how it affects their predicted outcome in real-time.")
    st.info("💡 Adjust the sliders to see instant prediction updates!")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider('Age at enrollment', 17, 70, 20)
        avg_grade = st.slider('Average Grade', 0.0, 20.0, 12.0, step=0.5)
        avg_success_rate = st.slider('Success Rate', 0.0, 1.0, 0.8, step=0.05)
    
    with col2:
        financial_issue = st.selectbox('Financial Issues?', [0, 1], key='wi_fi', 
                                      format_func=lambda x: "No" if x == 0 else "Yes")
        scholarship_holder = st.selectbox('Scholarship Holder?', [0, 1], key='wi_sh',
                                         format_func=lambda x: "No" if x == 0 else "Yes")
        gender = st.selectbox('Gender', [0, 1], key='wi_ge',
                            format_func=lambda x: "Female" if x == 0 else "Male")
    
    with col3:
        attendance = st.selectbox('Attendance Type', [1, 0], key='wi_at',
                                 format_func=lambda x: "Daytime" if x == 1 else "Evening")
        units_1st = st.slider('Units 1st Semester', 0, 30, 6)
        units_2nd = st.slider('Units 2nd Semester', 0, 30, 6)

    # Real-time Prediction
    feature_names = data["X_train"].columns
    user_input_df = pd.DataFrame([[
        age, avg_grade, avg_success_rate, financial_issue, scholarship_holder, 
        gender, attendance, units_1st, units_2nd
    ]], columns=feature_names)

    # Process input
    selector = SelectKBest(score_func=f_classif, k=8)
    selector.fit(data["X_train"], data["y_train"])
    processed_input = selector.transform(user_input_df)

    # Get prediction
    simple_model = models["Simple Model (9 features)"]
    prediction = simple_model.predict(processed_input)[0]
    probabilities = simple_model.predict_proba(processed_input)[0]

    # Display Live Prediction
    st.divider()
    st.subheader("⚡ Live Prediction Update")
    
    col1, col2, col3 = st.columns(3)
    predicted_label = TARGET_MAP.get(prediction, 'Unknown')
    
    with col1:
        if prediction == 0:
            st.metric("Outcome", predicted_label, delta="🚨 Dropout Risk", delta_color="inverse")
        elif prediction == 1:
            st.metric("Outcome", predicted_label, delta="⏳ Continuing", delta_color="off")
        else:
            st.metric("Outcome", predicted_label, delta="🎓 Graduate Path", delta_color="normal")
    
    with col2:
        st.metric("Confidence", f"{max(probabilities)*100:.1f}%")
    
    with col3:
        st.metric("Model", "Simple Model", delta="9 Features")

    # Probability Chart
    st.subheader("📊 Live Prediction Probabilities")
    prob_df = pd.DataFrame({
        'Outcome': [TARGET_MAP[i] for i in [0, 1, 2]],
        'Probability': probabilities * 100
    })

    fig_prob = px.bar(prob_df, x='Outcome', y='Probability', 
                     color='Outcome', title="Real-time Outcome Probabilities",
                     color_discrete_map={'Dropout': '#E97121', 'Enrolled': '#2196F3', 'Graduate': '#4CAF50'},
                     text_auto='.1f')
    fig_prob.update_yaxes(range=[0, 100])
    fig_prob.update_layout(showlegend=False)
    st.plotly_chart(fig_prob, use_container_width=True)