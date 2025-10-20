import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, classification_report

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"\n {model_name}")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(classification_report(y_test, y_pred))
    return model, acc, f1

def main():
    print(" Starting improved training (sklearn only)...")

    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/y_test.csv").values.ravel()

    #  Step 1: Feature Selection
    selector = SelectKBest(score_func=f_classif, k=8)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    #  Step 2: Logistic Regression (with balanced weights)
    log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
    log_reg, acc_lr, f1_lr = train_and_evaluate(log_reg, X_train, X_test, y_train, y_test, "Logistic Regression (Balanced)")

    # 🌲 Step 3: Random Forest with Grid Search
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [6, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3],
    }

    grid = GridSearchCV(rf, param_grid, scoring='f1_weighted', cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_
    print(f"\n🔥 Best RF Params: {grid.best_params_}")
    best_rf, acc_rf, f1_rf = train_and_evaluate(best_rf, X_train, X_test, y_train, y_test, "Random Forest (Tuned)")

    # 🏆 Choose best model
    best_model = best_rf if f1_rf > f1_lr else log_reg
    joblib.dump(best_model, "models/model_optimized.pkl")

    print(f"\n Best Model Saved — F1: {max(f1_lr, f1_rf):.4f}")

if __name__ == "__main__":
    main()
