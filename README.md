# LoanLens

### Explainable AI Credit Decision Simulator

LoanLens is an explainable machine learning system that predicts whether a loan application is likely to be **Approved** or **Rejected**, explains the factors influencing the prediction, and explores realistic actionable changes that could potentially alter the model's decision.

The project was developed for the **AI for Engineers (UCS321)** mini-project based on the HDFC Bank loan approval problem statement.

> **Important:** LoanLens is an educational machine learning project. It does not represent HDFC Bank's actual loan approval system, policies, underwriting criteria, or decision-making process.

---

## Overview

Traditional classification models can produce a prediction without explaining why that prediction was made. LoanLens addresses this limitation by combining:

- **Supervised machine learning**
- **Model evaluation and comparison**
- **SHAP-based explainability**
- **Actionable counterfactual analysis**
- **FastAPI backend**
- **REST API endpoints for integration with a frontend**

For a given loan application, LoanLens can answer three questions:

1. **What does the model predict?**
2. **Why did the model make that prediction?**
3. **What realistic changes to actionable application features could potentially change the model's prediction?**

---

## Problem Statement

The project problem statement asks students to develop a supervised machine learning classification model for HDFC Bank Ltd. that categorizes loan applications as:

- **Approved**
- **Rejected**

based on applicant information such as:

- Applicant income
- Coapplicant income
- Loan amount
- Loan term
- Credit history
- Employment status
- Property area
- Education
- Marital status
- Dependents
- Age

LoanLens extends the basic classification task by adding **model explainability and counterfactual analysis**.

---

## Key Features

### 1. Loan Approval Prediction

A trained classification pipeline predicts whether an application is:

```text
Approved
```

or

```text
Rejected
```

The API also returns the model-estimated probability of approval.

---

### 2. SHAP Explainability

LoanLens uses **SHAP (SHapley Additive exPlanations)** to explain individual predictions.

For every application, the system identifies the features that contributed most strongly to the model's decision.

For example:

```text
Loan Amount          +1.62
Coapplicant Income   +1.12
Applicant Income     +0.61
Education            -0.18
```

Positive contributions support the model's approval score, while negative contributions work against it.

> SHAP values represent contributions to the model's decision function and should not be interpreted as percentages or causal effects.

---

### 3. Actionable Counterfactual Analysis

LoanLens goes beyond simply explaining a prediction.

It searches for realistic hypothetical changes that could potentially change the model's decision.

For example:

```text
Current:
Applicant Income = ₹48,000
Prediction = Rejected

Counterfactual:
Applicant Income = ₹63,000
Predicted outcome = Approved
```

The system considers changes to actionable financial variables such as:

- Applicant income
- Coapplicant income
- Loan amount
- Loan term

`Credit_History` is deliberately excluded from counterfactual modification because the project treats it as a non-actionable input.

The counterfactual engine:

- Searches individual-feature changes first
- Searches combinations of actionable features when necessary
- Restricts candidate values to realistic ranges derived from the available dataset
- Penalizes larger changes through a normalized change-cost metric
- Returns the lowest-cost successful scenario when possible

If no realistic scenario changes the model's prediction, LoanLens explicitly reports that instead of inventing a recommendation.

> Counterfactual results represent hypothetical behavior of the trained model. They are not guarantees of loan approval, bank requirements, or financial advice.

---

## Machine Learning Approach

### Data

The project uses a publicly available synthetic loan approval dataset from Kaggle.

Dataset source:

https://www.kaggle.com/datasets/sanjamw/loan-approval-prediction-dataset

The dataset contains applicant and loan information including:

- Gender
- Married
- Dependents
- Education
- Employment Status
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area
- Age
- Loan Status

The dataset is **synthetic** and is intended for educational/modeling purposes. It should not be presented as real HDFC Bank customer data.

---

## Data Preprocessing

The production model uses a scikit-learn preprocessing pipeline.

### Numerical features

Numerical variables are processed using:

1. Median imputation
2. Standardization using `StandardScaler`

### Categorical features

Categorical variables are processed using:

1. Most-frequent-value imputation
2. One-hot encoding

Unknown categorical values are handled using:

```python
OneHotEncoder(handle_unknown="ignore")
```

This preprocessing is integrated directly into the trained model pipeline.

---

## Model Evaluation

Multiple classification algorithms were evaluated during model development, including:

- Logistic Regression
- Random Forest
- Decision Tree

The final LoanLens production model uses **Logistic Regression**.

The choice was not based solely on maximizing accuracy. Logistic Regression provides:

- Strong classification performance on the dataset
- A transparent model structure
- Compatibility with SHAP-based explanations
- A convenient decision function for counterfactual analysis
- A relatively lightweight production artifact

The model achieved approximately **98% test accuracy** during evaluation.

Because the dataset is synthetic and highly dependent on certain features, performance should not be interpreted as evidence of real-world banking performance.

---

## Feature Importance and Ablation

An important observation during model development was the strong influence of `Credit_History`.

Feature ablation experiments showed that removing `Credit_History` caused a substantial deterioration in predictive performance.

This finding is useful because it demonstrates that a high-performing model can still require careful interpretation.

LoanLens therefore exposes the model's reasoning rather than presenting its prediction as an unexplained result.

---

## Decision Strength

In addition to the predicted class and approval probability, the API calculates a **decision margin**:

```text
Decision Margin = |Approval Probability - 0.5|
```

The margin is mapped to a qualitative decision strength:

| Decision Margin | Strength |
|---:|---|
| ≥ 0.40 | Very Strong |
| ≥ 0.20 | Strong |
| ≥ 0.10 | Moderate |
| < 0.10 | Borderline |

This describes how far the model's estimated approval probability is from the classification threshold.

> Decision strength should not be interpreted as calibrated confidence or certainty.

---

# System Architecture

```text
                    ┌──────────────────────┐
                    │   Loan Application   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   FastAPI Backend    │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐
       │ Prediction  │  │    SHAP     │  │ Counterfactual │
       │    Model    │  │ Explanation │  │     Engine      │
       └──────┬──────┘  └──────┬──────┘  └────────┬────────┘
              │                │                  │
              └────────────────┼──────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Complete Analysis    │
                    │      Response        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Frontend        │
                    │  Analysis Dashboard  │
                    └──────────────────────┘
```

---

# API

LoanLens provides a REST API using **FastAPI**.

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

---

## Prediction

```http
POST /predict
```

Returns:

- Predicted class
- Approval probability

---

## Explainability

```http
POST /explain
```

Returns SHAP-based feature contributions for the submitted application.

---

## Counterfactual Analysis

```http
POST /counterfactual
```

Searches for realistic actionable changes that could potentially alter the model prediction.

---

## Complete Analysis

```http
POST /analyze
```

This is the primary endpoint intended for frontend integration.

It combines:

- Prediction
- Approval probability
- Decision margin
- Decision strength
- SHAP explanations
- Counterfactual scenarios
- Counterfactual status
- Best tested probability when no successful counterfactual is found

A simplified response structure is:

```json
{
  "prediction": "Rejected",
  "approval_probability": 0.372,
  "decision_margin": 0.128,
  "decision_strength": "Moderate",
  "explanations": [],
  "counterfactuals": [],
  "counterfactual_found": true
}
```

---

# Project Structure

```text
loanlens/
│
├── backend/
│   ├── __init__.py
│   │
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       ├── schemas.py
│       │
│       ├── routes/
│       │   ├── __init__.py
│       │   ├── predict.py
│       │   ├── explain.py
│       │   ├── counterfactual.py
│       │   └── analyze.py
│       │
│       └── services/
│           ├── __init__.py
│           ├── model_service.py
│           ├── shap_service.py
│           └── counterfactual_service.py
│
├── frontend/
│
├── data/
│   ├── raw/
│   │   └── loan_approval.csv
│   └── processed/
│
├── models/
│   └── loanlens_logistic_pipeline.joblib
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_explainability_complete.ipynb
│   └── 04_counterfactual_engine.ipynb
│
├── tests/
│   └── test_api.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/utkarsh-goel18/loanlens.git
cd loanlens
```

## 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Setup

The raw dataset is intentionally **not included in the Git repository**.

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/sanjamw/loan-approval-prediction-dataset

Place the CSV file at:

```text
data/raw/loan_approval.csv
```

The backend expects this file for SHAP background data and counterfactual range calculations.

The trained production model is included in the repository because it is a small deployment artifact required to run the API.

---

# Running the Backend

From the project root:

```bash
uvicorn backend.app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI automatically provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

and alternative documentation at:

```text
http://127.0.0.1:8000/redoc
```

---

# Testing

LoanLens includes API tests using `pytest`.

Run:

```bash
pytest
```

The test suite covers:

- Health endpoint
- Prediction endpoint
- Explainability endpoint
- Counterfactual endpoint
- Complete analysis endpoint
- Input validation for invalid credit-history values

Current test suite:

```text
6 passed
```

---

# Input Validation

The API validates incoming loan applications using Pydantic.

Examples of enforced constraints include:

```text
Applicant_Income >= 0
Coapplicant_Income >= 0
Loan_Amount >= 0
Loan_Term > 0
Credit_History ∈ {0, 1}
Age > 0 and <= 100
```

Invalid requests are rejected before reaching the model.

---

# Design Decisions

## Why Logistic Regression?

The project evaluates multiple classification approaches, but Logistic Regression was selected for the production system because LoanLens is designed around **interpretability as well as prediction**.

The goal is not simply:

```text
Input → Prediction
```

but:

```text
Input
  ↓
Prediction
  ↓
Why?
  ↓
What could potentially change it?
```

A model that performs well while remaining easier to interpret is therefore appropriate for this project.

---

## Why SHAP?

Accuracy alone does not explain an individual prediction.

SHAP allows LoanLens to expose the contribution of individual features to the model's decision function, making the prediction easier to inspect and communicate.

---

## Why Counterfactuals?

Feature importance answers:

> "Why did the model make this decision?"

Counterfactual analysis answers:

> "What hypothetical changes could potentially alter this decision?"

Combining both provides a more complete explanation of the model's behavior.

---

# Limitations

LoanLens has several important limitations.

### 1. Synthetic Dataset

The dataset is synthetic and should not be treated as real HDFC Bank customer data.

### 2. Model Dependency

Counterfactuals describe what the **trained model** would predict for a modified input. They do not represent actual bank policy.

### 3. Non-Causal Explanations

SHAP values describe model behavior and feature contributions. They do not establish causal relationships.

### 4. Credit History

Credit history has a strong influence on the model and is intentionally treated as non-actionable in the counterfactual engine.

### 5. Probability Interpretation

The approval probability is the model's estimated probability. It should not automatically be interpreted as a calibrated probability of receiving a real loan.

### 6. Real-World Deployment

A real credit decision system would require substantially more work, including appropriate financial data, regulatory compliance, fairness analysis, model governance, monitoring, security, and human oversight.

---

# Ethical Considerations

Credit decision systems can have significant consequences for individuals.

LoanLens is therefore presented as an **educational explainability and simulation project**, not as an automated real-world lending system.

Important considerations for a production credit system would include:

- Fairness and bias evaluation
- Data privacy
- Explainability
- Regulatory compliance
- Human review
- Model monitoring
- Security
- Robust validation on representative real-world data

---

# Future Improvements

Potential future extensions include:

- Fairness and bias analysis
- Calibration analysis
- Model monitoring
- More advanced counterfactual optimization
- Constraint-aware counterfactual generation
- Automated model comparison
- Feature interaction analysis
- Authentication and production-grade API security
- Deployment using a cloud platform
- Improved frontend visualization

These are outside the current project scope.

---

# Technologies Used

### Machine Learning

- Python
- pandas
- NumPy
- scikit-learn
- SHAP
- joblib

### Backend

- FastAPI
- Pydantic
- Uvicorn

### Testing

- pytest

### Frontend

- React

---

# Project Status

| Component | Status |
|---|---|
| Data preprocessing | Complete |
| Exploratory data analysis | Complete |
| Baseline model comparison | Complete |
| Logistic Regression production model | Complete |
| SHAP explainability | Complete |
| Actionable counterfactual engine | Complete |
| FastAPI backend | Complete |
| API validation tests | Complete |
| Frontend integration | In progress |
| Deployment | Future work |

---

# Academic Context

**Course:** AI for Engineers (UCS321)

**Problem:** HDFC Bank Loan Approval Classification

The original problem statement requires development of a supervised machine learning classification model that categorizes loan applications as Approved or Rejected based on applicant and financial information.

LoanLens extends the required classification task with explainability and counterfactual analysis.

---

# Disclaimer

LoanLens is an academic and educational project.

The dataset used by this project is synthetic and does not represent actual HDFC Bank customer records or underwriting decisions.

Predictions, probabilities, SHAP explanations, and counterfactual scenarios are outputs of a machine learning model trained on the selected dataset. They should not be interpreted as actual bank decisions, guaranteed loan outcomes, financial advice, or recommendations for real credit applications.

---

# Author

**Utkarsh Goel**

GitHub:  
https://github.com/utkarsh-goel18
