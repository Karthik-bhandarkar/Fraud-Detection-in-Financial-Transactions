# Fraud-Detection-in-Financial-Transactions
End-to-end Fraud Detection in Financial Transactions: ML pipeline + Streamlit frontend. Trains on a Kaggle dataset (6,362,620 rows, 11 features) with 94% accuracy; includes data preprocessing, modeling, evaluation, and a deployable UI. Reproducible environment.
# Fraud Detection in Financial Transactions: End-to-End ML Pipeline + Streamlit Frontend

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: ≥3.9](https://img.shields.io/badge/Python-≥3.9-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-blue.svg)](https://streamlit.io/)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com)

Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Key Features & Impact](#key-features--impact)
- [Quick Start (Local Run)](#quick-start-local-run)
- [File & Directory Structure](#file--directory-structure)
- [Modeling & Evaluation](#modeling--evaluation)
- [Streamlit App Demo](#streamlit-app-demo)
- [Reproducibility & Testing](#reproducibility--testing)
- [Licensing & Data Licensing](#licensing--data-licensing)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

<details>
<summary>Overview</summary>

- End-to-end fraud detection workflow implemented on a Kaggle dataset with 6,362,620 rows and 11 columns.
- Data exploration, preprocessing, model training (imbalanced classification with class_weight='balanced'), evaluation, and a deployable UI built with Streamlit.
- Model performance: 94% accuracy on the test set (reference to video/tutorial results; you can adjust metrics if you used a different model).
- Deliverables include reproducible code, a Python-based ML pipeline, and an interactive web interface for real-time predictions.

</details>

<details>
<summary>Dataset</summary>

- Source: Kaggle fraud detection dataset (link in the description on Kaggle).
- Size: 6,362,620 rows, 11 features (example features include: step, type, amount, name original, old balance, new balance, name destination, old balance destination, new balance destination, is fraud, is flagged fraud).
- Target: is_fraud (binary: 1 = fraud, 0 = not fraud).
- Licensing: Kaggle dataset terms apply. Do not redistribute the raw dataset; provide a link and instructions to download per Kaggle terms.
</details>

<details>
<summary>Tech Stack</summary>

- Language: Python 3.9+
- Data: pandas, NumPy
- ML: scikit-learn (ColumnTransformer, Pipeline, OneHotEncoder, StandardScaler, LogisticRegression with class_weight='balanced')
- UI: Streamlit
- Persistence: joblib
- Visualization: matplotlib, seaborn (EDA)
- Ecosystem: virtualenv/venv, Git, GitHub
</details>

<details>
<summary>Key Features & Impact</summary>

- End-to-end ML pipeline (data preprocessing, feature engineering, model training, evaluation)
- Handling class imbalance with balanced class weights
- Reproducible train/test split (70/30, stratified)
- Persisted trained model via joblib and a Streamlit frontend for real-time predictions
- Clear separation of concerns: data processing, modeling, and UI
- Lightweight deployment pathway: local Streamlit app with a single command

</details>

<details>
<summary>Quick Start (Local Run)</summary>

Prerequisites:
- Python 3.9+
- Virtual environment tool (venv or virtualenv)

Steps:
1) Create and activate a virtual environment
- macOS/Linux:
  python3 -m venv venv
  source venv/bin/activate
- Windows:
  python -m venv venv
  venv\Scripts\activate

2) Install dependencies
- pip install -r requirements.txt

3) Prepare data
- Download the Kaggle fraud dataset (link in README) and place it under data/ (or configure the path in code).

4) Train the model
- python -m src.train_pipeline  # or your actual training script

5) Run the Streamlit app
- streamlit run app/fraud_detection.py

6) Open the app
- Navigate to http://localhost:8501 in your browser

Note: If you’re using Docker or a containerized workflow, adapt commands accordingly.

</details>

<details>
<summary>File & Directory Structure</summary>

fraud-detection-project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── train_pipeline.py  # train and evaluate the model; save pipeline
│   ├── preprocess.py       # (optional) data cleaning / feature engineering
│   └── model.py            # (optional) model definition
├── app/
│   └── fraud_detection.py   # Streamlit UI
├── models/
│   └── fraud_detection_pipeline.pickle
├── tests/
├── requirements.txt
├── README.md
└── LICENSE
</details>

<details>
<summary>Modeling & Evaluation</summary>

- Features:
  - Numerical: step, amount, old_balance, new_balance, old_balance_destination, new_balance_destination
  - Categorical: type
- Pipeline:
  - ColumnTransformer with StandardScaler (numerical) and OneHotEncoder (categorical)
  - LogisticRegression with class_weight='balanced'
- Evaluation:
  - Train/test split: 70/30 with stratification on is_fraud
  - Metrics: classification_report, confusion_matrix
  - Reported accuracy: ~94% on the test set (adjust if you used a different model or data)
- Persistence:
  - Pipeline saved as fraud_detection_pipeline.pickle via joblib
</details>

<details>
<summary>Streamlit App Demo</summary>

- fraud_detection.py loads the saved pipeline and renders a simple UI:
  - Inputs: transaction type, amount, old/new balances (sender/destination)
  - Predict button displays the result as Fraud / Not Fraud with color-coded feedback
- Local testing: run via streamlit run app/fraud_detection.py
- Optional: wire up a small demo dataset for quick tests

</details>

<details>
<summary>Reproducibility & Testing</summary>

- Random seed: set a fixed seed (e.g., 42) for train/test split and model initialization
- Requirements: pin versions in requirements.txt
- Tests: optional unit tests for preprocessing and prediction path
- Data handling: clearly document how to obtain and reference Kaggle data, with license notes
</details>

<details>
<summary>Licensing & Data Licensing</summary>

- License: MIT (or Apache-2.0; see License section)
- Data licensing: Kaggle terms apply; do not commit the raw dataset
- If you publish a Python package, include license metadata (pyproject.toml or setup.py)
</details>

<details>
<summary>Contributing</summary>

- Fork the repository
- Create a feature branch (e.g., feat/streamlit-ui)
- Open a pull request with a short description of changes
- Follow PEP8/CODE-STYLE guidelines
- Include tests or documentation where applicable
</details>

<details>
<summary>License</summary>

- This project is licensed under the MIT License. See LICENSE for details.
</details>

<details>
<summary>Acknowledgments</summary>

- Kaggle: Fraud Detection Dataset
- Tutorial inspiration: Fraud Detection with ML and Streamlit
- Community contributions and practice in end-to-end ML pipelines
</details>

<details>
<summary>Contact</summary>

- Author: [Karthik Bhandarkar]
- Email: [karthikbhandarkar2004@gmail.com]
- LinkedIn: [www.linkedin.com/in/karthik-bhandarkar]
- GitHub: [https://github.com/Karthik-bhandarkar]
</details>

