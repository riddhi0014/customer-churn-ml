# Customer Churn Prediction (Machine Learning)

## Project Overview

This project builds a binary classification model to predict customer churn in a telecommunications business.
The objective is to identify customers who are likely to leave so the business can take proactive retention actions.

The model uses customer demographics, service usage, account details, and billing information to estimate churn probability.

## Problem Statement

Customer churn creates significant revenue loss in subscription-based businesses.
By identifying high-risk customers early, teams can apply targeted interventions such as personalized offers or service improvements.

Project goals:

- Build a model to predict churn (`Yes` / `No`)
- Evaluate performance with classification metrics
- Prioritize churner detection (high recall)
- Align predictions with business cost trade-offs

## Dataset

Source: Telco Customer Churn dataset (`data/telco_churn.csv`)

Features include:

- Customer demographics (`gender`, `SeniorCitizen`, `Dependents`, etc.)
- Account details (`tenure`, `Contract`, `PaymentMethod`)
- Subscribed services (`InternetService`, `StreamingTV`, `OnlineSecurity`, etc.)
- Billing information (`MonthlyCharges`, `TotalCharges`)
- Target variable: `Churn` (`Yes` / `No`)

## Data Preprocessing

Preprocessing is centralized in `src/data_preprocessing.py` to keep training and evaluation consistent.

Steps:

- Convert `TotalCharges` to numeric and impute missing values with median
- Remove non-informative identifier (`customerID`)
- Apply one-hot encoding to categorical features (`drop_first=True`)
- Encode target: `No -> 0`, `Yes -> 1`

## Model Selection

Models evaluated:

- Logistic Regression
- Gradient Boosting

Both achieved similar ROC-AUC (~0.845).
Logistic Regression was selected because it provides comparable performance, simpler interpretation, and stable optimization.

Class imbalance handling:

- `class_weight="balanced"`

## Threshold Tuning

Instead of using the default threshold (`0.50`), prediction threshold was tuned to `0.55` to improve F1-score while maintaining high recall for churners.

This supports the business goal that missing a true churner is more costly than a false positive.

## Final Model Performance

Test set results:

- Accuracy: `75.4%`
- Recall (Churn): `76%`
- Precision (Churn): `53%`
- F1-score (Churn): `0.62`
- ROC-AUC: `0.84`
- Cross-validated AUC: `0.845`

## Project Structure

```text
customer-churn-ml-aiproject/
├── data/
│   └── telco_churn.csv
├── models/
├── notebooks/
│   └── churn_analysis.ipynb
├── results/
├── src/
│   ├── data_preprocessing.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── train_model.py
└── README.md
```

## How to Run

From the project root:

1. Train the model

```bash
python src/train_model.py
```

This will:

- Load and preprocess the data
- Train the Logistic Regression pipeline
- Save the model bundle to `models/churn_model.pkl` (includes model + threshold)

2. Evaluate the model

```bash
python src/evaluate_model.py
```

This will:

- Print accuracy, ROC-AUC, and classification report
- Generate and save confusion matrix at `results/confusion_matrix.png`

## Business Insight

The model is tuned for strong churner identification (high recall).
By correctly identifying around 76% of churners, the business can apply targeted retention actions and reduce potential revenue loss.

## Notes

- `src/predict.py` is currently an empty placeholder and can be implemented for single/batch inference.

## Run in Google Colab

This project can be executed directly in Google Colab.

The Colab notebook:

- Clones this GitHub repository
- Installs dependencies
- Trains the model
- Evaluates performance
- Displays classification metrics and confusion matrix

Repository link used in the notebook:

https://github.com/riddhi0014/customer-churn-ml

Google Colab notebook link:

https://colab.research.google.com/drive/13b7UEL8m31v9AeGKW1ctmfJZlMmo2_9_?usp=sharing

Open the notebook and select **Runtime → Run All** to execute the complete pipeline.

## Conclusion

This project demonstrates a structured, reproducible, and business-aligned machine learning workflow for customer churn prediction, incorporating class balancing, threshold tuning, cross-validation, and modular engineering practices.