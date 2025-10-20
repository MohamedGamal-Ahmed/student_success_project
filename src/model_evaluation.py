"""
Model Evaluation Script
-----------------------
This script loads the trained model and evaluates its performance
on the test data using standard classification metrics.
"""

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
os.makedirs("reports", exist_ok=True)


def evaluate_model():
    print(" Starting model evaluation...")

    # ---------------------------
    # Load data and model
    # ---------------------------
    try:
        X_test = pd.read_csv("data/X_test.csv")
        y_test = pd.read_csv("data/y_test.csv")
        model = joblib.load("models/model.pkl")
        print(" Data and model loaded successfully!")
    except Exception as e:
        print(f" Error loading data or model: {e}")
        return

    # ---------------------------
    # Make predictions
    # ---------------------------
    y_pred = model.predict(X_test)

    # ---------------------------
    # Evaluate metrics
    # ---------------------------
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n Model Evaluation Results:")
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    # ---------------------------
    # Save evaluation report
    # ---------------------------
    with open("reports/model_evaluation.txt", "w") as f:
        f.write("Model Evaluation Report\n")
        f.write("=======================\n\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print("\n Evaluation report saved to: reports/model_evaluation.txt")

# Run the evaluation
if __name__ == "__main__":
    evaluate_model()
