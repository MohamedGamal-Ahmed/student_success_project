import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import os

def train_optimized_random_forest():
    print(" Starting Random Forest Optimization...\n")

    # Load data
    df = pd.read_csv("data/clean_students.csv")
    X = df.drop(columns=["Target"])
    y = df["Target"]

    # Encode target
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define Random Forest + parameters grid
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced']
    }

    # Grid Search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='f1_weighted',
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    # Best model
    best_rf = grid_search.best_estimator_
    print(f" Best Parameters: {grid_search.best_params_}\n")

    # Evaluate
    y_pred = best_rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f" Optimized Random Forest Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_rf, "models/random_forest_optimized.pkl")
    print("\n Model saved successfully: models/random_forest_optimized.pkl")

if __name__ == "__main__":
    train_optimized_random_forest()
