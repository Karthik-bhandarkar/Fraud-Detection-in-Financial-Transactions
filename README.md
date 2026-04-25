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

