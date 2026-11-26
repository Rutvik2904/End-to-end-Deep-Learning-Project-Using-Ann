# Customer Churn Prediction (ANN)

An end-to-end Deep Learning web application that predicts whether a bank customer is likely to exit (churn) based on demographic and account metrics. Built using an Artificial Neural Network (ANN) in TensorFlow/Keras and deployed via Streamlit.

## Project Structure

```text
├── app.py                      # Streamlit web application for real-time predictions
├── Churn_Modelling_India.csv   # Dataset containing customer features and churn labels
├── experiments.ipynb           # Data preprocessing, feature encoding, and ANN model training
├── prediction.ipynb            # Standalone inference testing notebook
├── model.keras                 # Saved trained Keras ANN model
├── label_encoder_gender.pkl    # Fitted LabelEncoder for the 'Gender' feature
├── onehot_encoder_geo.pkl      # Fitted OneHotEncoder for the 'Geography' feature
├── scaler.pkl                  # Fitted StandardScaler for numerical feature normalization
└── requirements.txt            # Python dependencies
```

## Features & Workflow

1. **Data Preprocessing (`experiments.ipynb`)**:
   - Encodes categorical variables (`Gender` via Label Encoding and `Geography` via One-Hot Encoding).
   - Scales numerical features (`CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`) using `StandardScaler`.
   - Serializes encoders and scaler as `.pkl` artifacts to prevent data leakage and ensure consistent inference transformations.
2. **Model Architecture (`model.keras`)**:
   - Feedforward Artificial Neural Network trained for binary classification (`Exited`: `0` = Retained, `1` = Churned).
3. **Interactive Web App (`app.py`)**:
   - Provides an intuitive UI to input customer profile details, applies the saved preprocessing pipeline, and outputs the churn probability alongside a clear retention/churn verdict.

## Installation & Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd <your-repository-folder>
   ```

2. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Streamlit Web App
Launch the interactive prediction interface locally:
```bash
streamlit run app.py
```

### Explore Notebooks
- Open `experiments.ipynb` in Jupyter Notebook or VS Code to inspect exploratory data analysis, model architecture, and training metrics.
- Open `prediction.ipynb` to test single-sample inference directly in Python.

## Input Features

| Feature | Type | Description |
| :--- | :--- | :--- |
| **Geography** | Categorical | Customer's region/location |
| **Gender** | Categorical | Male / Female |
| **Age** | Numerical | Customer's age in years |
| **Credit Score** | Numerical | Customer's credit score |
| **Tenure** | Numerical | Number of years with the bank |
| **Balance** | Numerical | Current account balance |
| **Num of Products** | Numerical | Number of bank products utilized |
| **Has Credit Card** | Binary | `1` if the customer holds a credit card, else `0` |
| **Is Active Member** | Binary | `1` if the customer is active, else `0` |
| **Estimated Salary** | Numerical | Customer's estimated annual salary |
