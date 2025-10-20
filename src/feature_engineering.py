import pandas as pd
from sklearn.model_selection import train_test_split
import os

def feature_engineering():
    print(" Starting feature engineering...")

    # Load cleaned data
    df = pd.read_csv("data/clean_students.csv")
    print(f"Data loaded successfully! Shape: {df.shape}")

    # ================================
    # 1 Create New Useful Features
    # ================================

    # Average grade between 1st and 2nd semester
    df["avg_grade"] = df[["Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]].mean(axis=1)

    # Average success rate (approved / enrolled) for both semesters
    df["success_rate_sem1"] = df["Curricular units 1st sem (approved)"] / df["Curricular units 1st sem (enrolled)"].replace(0, 1)
    df["success_rate_sem2"] = df["Curricular units 2nd sem (approved)"] / df["Curricular units 2nd sem (enrolled)"].replace(0, 1)
    df["avg_success_rate"] = df[["success_rate_sem1", "success_rate_sem2"]].mean(axis=1)

    # Indicator for financial difficulty (combining Debtor + Tuition not updated)
    df["financial_issue"] = ((df["Debtor"] == 1) | (df["Tuition fees up to date"] == 0)).astype(int)

    # ================================
    # 2 Select Features & Target
    # ================================
    X = df[[
        "Age at enrollment", "avg_grade", "avg_success_rate",
        "financial_issue", "Scholarship holder", "Gender", 
        "Daytime/evening attendance", "Curricular units 1st sem (enrolled)",
        "Curricular units 2nd sem (enrolled)"
    ]]
    y = df["Target"]

    # ================================
    # 3️⃣ Split Data
    # ================================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f" Data split done — Train: {X_train.shape}, Test: {X_test.shape}")

    # ================================
    # 4 Save Processed Data
    # ================================
    os.makedirs("data", exist_ok=True)
    final_df = pd.concat([X, y], axis=1)
    final_df.to_csv("data/final_students.csv", index=False)

    print(" Final processed dataset saved to: data/final_students.csv")
    print(" Feature engineering completed successfully!")

    # ✅ Save train/test splits for later evaluation
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    print(" Train/Test splits saved successfully to 'data/' folder.")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = feature_engineering()
