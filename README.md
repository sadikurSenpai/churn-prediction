## Customer Churn Prediction App

A simple **machine learning web app** built with **Streamlit** to predict whether a customer is likely to churn based on behavioral and demographic features.
The app uses a **Random Forest model** trained on structured customer data, with preprocessing (scaling + encoding) applied through a saved `ColumnTransformer`.

---

###  Features

* Interactive **Streamlit web UI** for real-time predictions
* Pretrained **Random Forest classifier** (`best_random_forest_model.pkl`)
* Consistent **data preprocessing pipeline** (`preprocessor.pkl`)
* Ready-to-run local setup with minimal dependencies

---

###  Project Structure

```
churn-prediction/
│
├── .venv/                        # Virtual environment (auto-generated)
├── app.py                        # Streamlit application
├── best_random_forest_model.pkl  # Trained Random Forest model
├── preprocessor.pkl              # Preprocessing pipeline (scaler + encoder)
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

###  Installation and Setup

#### 1️⃣ Clone the repository

```bash
git clone https://github.com/sadikurSenpai/churn-prediction.git
cd churn-prediction
```

#### 2️⃣ Create and activate a virtual environment

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

#### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 🖥️ Run the Streamlit app

After activating your virtual environment:

```bash
streamlit run app.py
```

Streamlit will open a local server (e.g., `http://localhost:8501`) where you can interact with the app.

---

### 🧾 How It Works

1. The app takes user input for customer attributes (e.g., Age, Income, Spend, Satisfaction).
2. Inputs are preprocessed using the saved `preprocessor.pkl`.
3. The trained Random Forest model (`best_random_forest_model.pkl`) predicts **churn probability**.
4. The result displays whether the customer is likely to **churn or stay**.

---

### 📦 Requirements

Main dependencies (see `requirements.txt`):

* `streamlit`
* `pandas`
* `numpy`
* `scikit-learn`
* `joblib`

Optional (if experimenting with training):

* `xgboost`
* `lightgbm`

---

### 🧰 Example Command Summary

```bash
git clone https://github.com/sadikurSenpai/churn-prediction.git
cd churn-prediction
python -m venv .venv
.venv\Scripts\activate     # or source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

### 📊 Model Info

* **Model type:** Random Forest Classifier
* **Best parameters:**
  `{'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 50, 'bootstrap': True}`
* **Preprocessing:** StandardScaler + OneHotEncoder (via ColumnTransformer)
* **Dataset size:** 1000 samples, balanced target variable