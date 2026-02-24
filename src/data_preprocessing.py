import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    # Fix TotalCharges datatype
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Separate target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Drop ID column
    X = X.drop("customerID", axis=1)

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Encode target
    y = y.map({"No": 0, "Yes": 1})

    return X, y