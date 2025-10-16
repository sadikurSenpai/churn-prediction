## Customer Churn Prediction App

A simple machine learning web app built with Streamlit to predict whether a customer is likely to churn based on behavioral and demographic features.  
The app uses a Random Forest model trained on structured customer data, with preprocessing (scaling and encoding) applied through a saved `ColumnTransformer`.

---

<img width="919" height="523" alt="image" src="https://github.com/user-attachments/assets/f24852df-c826-4f61-9107-c76a83845570" />

---

<img width="920" height="518" alt="image" src="https://github.com/user-attachments/assets/0f8a83da-5f13-4076-827d-2f3b2b5569fd" />



### Features

* Interactive Streamlit web UI for real-time predictions  
* Pretrained Random Forest classifier (`best_random_forest_model.pkl`)  
* Consistent data preprocessing pipeline (`preprocessor.pkl`)  
* Ready-to-run local setup with minimal dependencies  

---

### Project Structure

```

churn-prediction/
│
├── .venv/                        # Virtual environment (auto-generated)
├── app.py                        # Streamlit application
├── best_random_forest_model.pkl  # Trained Random Forest model
├── preprocessor.pkl              # Preprocessing pipeline (scaler + encoder)
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation

````

---

### Dataset Description

The dataset contains **1,000 customer records** with a mix of demographic, behavioral, and engagement attributes.  
Key features include:

- **Age, Annual_Income, Total_Spend, Num_of_Purchases, Satisfaction_Score, Last_Purchase_Days_Ago**, etc.  
- Categorical attributes such as **Gender**, **Email_Opt_In**, and **Promotion_Response**.  
- Target variable: **Target_Churn** (binary; 1 = churned, 0 = stayed).  
- The dataset is **balanced** with approximately 52.6% churned and 47.4% retained customers.

---

### Installation and Setup

#### 1. Clone the repository

```bash
git clone https://github.com/sadikurSenpai/churn-prediction.git
cd churn-prediction
````

#### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### Run the Streamlit App

After activating your virtual environment:

```bash
streamlit run app.py
```

Streamlit will open a local server (e.g., `http://localhost:8501`) where you can interact with the app.

---

### How It Works

1. The app takes user input for customer attributes (e.g., Age, Income, Spend, Satisfaction).
2. Inputs are preprocessed using the saved `preprocessor.pkl`.
3. The trained Random Forest model (`best_random_forest_model.pkl`) predicts churn probability.
4. The result displays whether the customer is likely to churn or stay.

---

### Requirements

Main dependencies (see `requirements.txt`):

* streamlit
* pandas
* numpy
* scikit-learn
* joblib

Optional (if experimenting with other models):

* xgboost
* lightgbm

---

### Example Command Summary

```bash
git clone https://github.com/sadikurSenpai/churn-prediction.git
cd churn-prediction
python -m venv .venv
.venv\Scripts\activate     # or source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

### Model Info

* **Model type:** Random Forest Classifier
* **Best parameters:**
  `{'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 50, 'bootstrap': True}`
* **Preprocessing:** StandardScaler + OneHotEncoder (via ColumnTransformer)
* **Dataset size:** 1,000 samples, balanced target variable

---

### Key Results

| Metric    | Random Forest |
| --------- | ------------- |
| Accuracy  | 0.4867        |
| Precision | 0.5116        |
| Recall    | 0.5569        |
| F1-Score  | 0.5333        |
| ROC-AUC   | 0.4628        |

Although the overall accuracy is moderate because of the lack of data, this implementation demonstrates a complete ML-to-app deployment pipeline with proper preprocessing, model tuning, and interface integration.

---
