
import pandas as pd
import joblib
import os

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

from data_preprocessing import load_data, preprocess_data
# # -----------------------------
# # 1. Load Data
# # -----------------------------
# def load_data(path):
#     df = pd.read_csv(path)
#     return df


# # -----------------------------
# # 2. Preprocess Data
# # -----------------------------
# def preprocess_data(df):
#     df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
#     df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

#     X = df.drop("Churn", axis=1)
#     y = df["Churn"]

#     X = X.drop("customerID", axis=1)
#     X = pd.get_dummies(X, drop_first=True)
#     y = y.map({"No": 0, "Yes": 1})

#     return X, y


# -----------------------------
# Main Evaluation
# -----------------------------
if __name__ == "__main__":

    data_path = "data/telco_churn.csv"
    model_path = "models/churn_model.pkl"

    # Load data
    df = load_data(data_path)
    X, y = preprocess_data(df)

    # Same split logic as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Load saved model
    saved = joblib.load(model_path)
    model = saved["model"]
    threshold = saved["threshold"]

    # Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/confusion_matrix.png")

    print("\nConfusion matrix saved to results/confusion_matrix.png")