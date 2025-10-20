import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os


print(" Starting Feature Importance Analysis...")

# Load training data (to get the same feature names used during training)
try:
    X_train = pd.read_csv("C:/Users/Mr-Mohamed/Ai_Amit_Diploma/student_success_project/data/X_test.csv")
    print(f" Data loaded successfully! Shape: {X_train.shape}")
except FileNotFoundError:
    print(" Error: X_train.csv not found in /data folder.")
    exit()

# Load the trained Random Forest model
try:
    model = joblib.load("C:/Users/Mr-Mohamed/Ai_Amit_Diploma/student_success_project/models/random_forest_optimized.pkl")
    print(" Model loaded successfully!")
except FileNotFoundError:
    print(" Error: Model file not found. Make sure 'random_forest_optimized.pkl' exists in /models.")
    exit()

# Check feature lengths
print(f"Features in X_train: {len(X_train.columns)}")
print(f"Importances in model: {len(model.feature_importances_)}")

# Match lengths if needed
min_len = min(len(X_train.columns), len(model.feature_importances_))

# Create DataFrame for feature importance
feature_importance = pd.DataFrame({
    "Feature": X_train.columns[:min_len],
    "Importance": model.feature_importances_[:min_len]
}).sort_values(by="Importance", ascending=False)

# Display Top 10 Features
print("\n Top 10 Most Important Features:")
print(feature_importance.head(10))

# Save report
# Ensure the reports directory exists
os.makedirs("reports", exist_ok=True)

# Save the report
feature_importance.to_csv("reports/feature_importance.csv", index=False)
print(" Feature importance report saved to reports/feature_importance.csv")

# Visualization
plt.figure(figsize=(10, 6))
plt.barh(feature_importance["Feature"].head(10), feature_importance["Importance"].head(10))
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Important Features (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
