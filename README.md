# 🧠 Credit Card Default Prediction

This project aims to predict whether a credit card customer will default on their payment next month using machine learning and deep learning models. It also includes model explainability using SHAP to understand feature contributions.

---


## 📊 Dataset

- Source: [UCI Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- Records: 30,000
- Features: Demographics, bill statements, repayment status
- Target: `default payment next month` (0 = No, 1 = Yes)

---

## 🔍 Notebooks Overview

### 1. `01_eda.ipynb` – Exploratory Data Analysis
- Data cleaning, renaming
- Visual analysis of target distribution
- Feature distribution and correlations

### 2. `02_model_Exploration.ipynb` – Model Training and Comparison
- Train/test split with `stratify`
- Feature scaling using `StandardScaler`
- Models tested:
  - Logistic Regression
  - Random Forest
  - SVM
  - KNN
  - XGBoost
  - ANN (TensorFlow)
- Evaluation metrics:
  - Accuracy, Precision, Recall, F1-score, ROC-AUC
- Best model selected based on F1-Score and AUC (ANN)

### 3. `03_explainability.ipynb` – SHAP Explainability
- SHAP with:
  - `KernelExplainer` for ANN
- Global feature importance and individual prediction insights

---

## 🏆 Key Highlights

- **Best Model:** ANN with highest F1-score for default class
- **SHAP Analysis:** Identifies key factors like payment delays and bill amounts
- **Scalability:** Ready for integration with Streamlit or FastAPI

---

## 🛠 Tech Stack

- Python 3.11+
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, XGBoost, TensorFlow/Keras
- SHAP
- Joblib (for model persistence)

---

## 💾 Saved Artifacts

| File                          | Description                      |
|-------------------------------|----------------------------------|
| `best_model_ann.h5`           | Best performing model (ANN)      |
| `scaler.pkl`                  | Scaler used during training      |

---

## 🚀 Future Improvements

- Add Streamlit UI for real-time predictions
- Hyperparameter tuning with Optuna or GridSearchCV
- Integrate MLflow for experiment tracking
- Deploy on cloud (AWS/GCP/Azure)

---

## 📬 Contact

For queries, reach out to the author via GitHub or LinkedIn.

---
