📌 Customer Churn Prediction – Machine Learning Project
📖 Project Overview

This project builds a classification-based machine learning model to predict customer churn in a telecommunications company. The goal is to identify customers who are likely to discontinue their subscription so that the company can take proactive retention actions.

The model uses customer demographic information, service usage patterns, billing details, and subscription history to estimate the probability of churn.

🎯 Problem Statement

Customer churn leads to significant revenue loss in subscription-based businesses. Accurately identifying customers at risk of churning allows companies to implement targeted retention strategies such as personalized offers or service improvements.

The objective of this project is to:

Build a classification model to predict churn (Yes/No)

Evaluate model performance using appropriate metrics

Optimize the model to prioritize identifying churners (high recall)

Align predictions with business cost considerations

📊 Dataset

The project uses the Telco Customer Churn Dataset, which includes:

Customer demographics (gender, senior citizen, dependents, etc.)

Account details (tenure, contract type, payment method)

Services subscribed (Internet, Streaming, Security, etc.)

Billing information (MonthlyCharges, TotalCharges)

Target variable: Churn (Yes/No)

⚙️ Data Preprocessing

The following preprocessing steps were applied:

Converted TotalCharges to numeric and handled missing values using median imputation

Removed non-informative features (customerID)

Applied one-hot encoding to categorical variables

Encoded target variable (Churn: Yes → 1, No → 0)

To avoid data leakage and ensure reproducibility, preprocessing was centralized in a shared module.

🤖 Model Selection

Several models were evaluated, including:

Logistic Regression

Gradient Boosting

Both models achieved similar ROC-AUC scores (~0.845). Logistic Regression was selected due to:

Comparable performance

Simpler and interpretable structure

Convex optimization with guaranteed global minimum

Easier business interpretability

Class imbalance was handled using:

class_weight="balanced"
🎚 Threshold Tuning

Instead of using the default classification threshold (0.5), the decision threshold was tuned to 0.55 to optimize F1-score while maintaining high recall for churn customers.

This approach aligns with business priorities, where missing a churner is more costly than incorrectly flagging a non-churner.

📈 Final Model Performance

On the test dataset:

Accuracy: 75.4%

Recall (Churn): 76%

Precision (Churn): 53%

F1-score (Churn): 0.62

ROC-AUC: 0.84

Cross-validated AUC: 0.845

The model demonstrates strong discriminatory power and effectively identifies high-risk customers.

📁 Project Structure
customer_churn_project/
│
├── data/                   # Dataset
├── models/                 # Saved trained model
├── notebooks/              # Exploratory analysis
├── results/                # Confusion matrix image
├── src/
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── data_preprocessing.py
└── README.md
🚀 How to Run the Project
1️⃣ Train the Model
python src/train_model.py

This will:

Load and preprocess data

Train the Logistic Regression model

Save the trained model to models/churn_model.pkl

2️⃣ Evaluate the Model
python src/evaluate_model.py

This will:

Print evaluation metrics

Display classification report

Save confusion matrix to results/confusion_matrix.png

🧠 Business Insight

The model prioritizes identifying customers at risk of churn by optimizing recall. By correctly identifying approximately 76% of churners, the company can significantly reduce revenue loss through proactive retention strategies.

📌 Conclusion

This project demonstrates a complete machine learning workflow:

Data preprocessing

Model training

Threshold tuning

Cross-validation

Performance evaluation

Reproducible pipeline design

The final model is interpretable, business-aligned, and suitable for deployment in churn prediction systems.
