import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
#     # Fix TotalCharges datatype
#     df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
#     df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

#     # Separate target
#     X = df.drop("Churn", axis=1)
#     y = df["Churn"]

#     # Drop ID column
#     X = X.drop("customerID", axis=1)

#     # One-hot encoding
#     X = pd.get_dummies(X, drop_first=True)

#     # Encode target
#     y = y.map({"No": 0, "Yes": 1})

#     return X, y


# -----------------------------
# 3. Train Model
# -----------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    pipeline.fit(X_train, y_train)

    return pipeline


# -----------------------------
# 4. Save Model
# -----------------------------
def save_model(model, threshold, path):
    joblib.dump({
        "model": model,
        "threshold": threshold
    }, path)


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    data_path = "data/telco_churn.csv"
    model_path = "models/churn_model.pkl"
    
    final_threshold = 0.55

    df = load_data(data_path)
    X, y = preprocess_data(df)
    model = train_model(X, y)
    save_model(model, final_threshold, model_path)

    print("Model training completed and saved successfully.")