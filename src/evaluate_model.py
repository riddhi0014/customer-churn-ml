
import pandas as pd
import joblib
import os

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

from data_preprocessing import load_data, preprocess_data


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

    # # Confusion Matrix
    # cm = confusion_matrix(y_test, y_pred)

    # plt.figure(figsize=(5,4))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.title("Confusion Matrix")
    
    # os.makedirs("results", exist_ok=True)
    # plt.savefig("results/confusion_matrix.png")

    # print("\nConfusion matrix saved to results/confusion_matrix.png")


    from sklearn.metrics import roc_curve, precision_recall_curve
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/confusion_matrix.png")
    print("\nConfusion matrix saved to results/confusion_matrix.png")
    
    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(roc_auc_score(y_test, y_prob)))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("results/roc_curve.png")
    print("\nROC curve saved to results/roc_curve.png")
    
    # Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig("results/precision_recall_curve.png")
    print("\nPrecision-Recall curve saved to results/precision_recall_curve.png")