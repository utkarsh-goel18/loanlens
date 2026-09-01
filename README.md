# LoanLens 🏦

### Explainable AI Credit Decision Simulator

> **Predict. Explain. Simulate.**

LoanLens is an AI-powered credit decision analysis system built around a loan approval classification problem. Instead of stopping at a simple **Approved / Rejected** prediction, LoanLens explores the reasoning and alternatives behind a model's decision.

The project combines machine learning, explainability, similarity search, and counterfactual analysis into a single engineering-focused system.

---

## 🎯 Problem

Traditional loan-approval ML projects generally answer one question:

> **Will this application be approved or rejected?**

LoanLens goes further:

- **What does the model predict?**
- **Why did it make that prediction?**
- **What happens if the applicant's circumstances change?**
- **How similar is this application to previous applications?**
- **How reliable is the model's prediction for this particular application?**

The project is based on the HDFC Bank loan approval problem statement provided for the **AI for Engineers** course.

---

## 💡 Core Features

### 1. Credit Approval Prediction

A supervised machine learning pipeline predicts the probability of a loan application being classified as **Approved** or **Rejected**.

Multiple classification algorithms will be trained and evaluated before selecting the final model.

### 2. Explainable Decisions

LoanLens does not treat the model as a black box. For each prediction, it identifies the applicant features that contribute most strongly to the decision.

The goal is to answer:

> **“Why did the model make this prediction?”**

### 3. Counterfactual Analysis — *What If?*

The signature feature of LoanLens.

For an application with a low predicted approval probability, the system explores changes to relevant applicant/loan variables and estimates how those changes affect the model's prediction.

For example:

```text
Current application
Approval probability: 34%

What if the requested loan amount is reduced?

₹25L  →  34%
₹22L  →  51%
₹20L  →  68%
₹18L  →  79%
```

The system therefore moves from simply **predicting a decision** to **exploring the model's decision boundary**.

> Counterfactual results represent model behavior and are not guarantees of real-world loan approval.

### 4. Similar Application Search

LoanLens retrieves historically similar applications from the dataset using a normalized feature representation and similarity measures.

This provides a case-based perspective alongside the model prediction:

```text
Current Application
        │
        ├── Similar Case A → Approved
        ├── Similar Case B → Approved
        ├── Similar Case C → Rejected
        └── Similar Case D → Approved
```

### 5. Prediction Reliability

A high probability does not automatically mean a prediction is trustworthy.

LoanLens will assess how closely a new application resembles the data used to train the model and flag unusual/out-of-distribution cases where appropriate.

Example:

> **Approval probability: 78%**  
> **Prediction reliability: Low**  
> This application is poorly represented by the observed training data.

---

## 🧠 Machine Learning Pipeline

```text
                Loan Dataset
                     │
                     ▼
              Data Preprocessing
                     │
                     ▼
                  EDA
                     │
                     ▼
            Feature Engineering
                     │
                     ▼
          Multiple Classification Models
                     │
                     ▼
             Model Evaluation
                     │
                     ▼
               Final Model
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
      Prediction  Explanation  Similarity
          │          │          │
          └──────────┼──────────┘
                     ▼
             Counterfactual Engine
                     │
                     ▼
             Reliability Analysis
                     │
                     ▼
             Credit Decision Interface
```

---

## 📊 Dataset

The current project uses a Kaggle **Loan Approval Prediction** dataset containing applicant, financial, employment, credit-history and property-related attributes with loan approval status as the target.

The dataset is **synthetic** and is used for educational and experimental purposes. LoanLens does not represent an actual HDFC Bank credit-scoring or lending system, and no claim is made that the model reflects HDFC Bank's real underwriting policy.

The raw dataset is kept locally under `data/raw/` and is excluded from version control.

---

## 🛠️ Technology Stack

### Machine Learning / Backend

- Python
- Pandas
- NumPy
- Scikit-learn
- SHAP / model explainability tools
- FastAPI

### Frontend

- React
- Tailwind CSS

### Development

- Git & GitHub
- Jupyter Notebook
- VS Code

---

## 📁 Project Structure

```text
loanlens/
│
├── data/
│   ├── raw/                 # Original datasets (local only)
│   └── processed/           # Cleaned/processed data
│
├── notebooks/               # Exploration and experiments
│
├── src/
│   ├── data/                # Data loading & preprocessing
│   ├── features/            # Feature engineering
│   ├── models/              # Training, evaluation & inference
│   ├── explainability/      # Model explanations
│   └── api/                 # Backend/API layer
│
├── tests/                   # Unit and integration tests
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚧 Project Status

**Currently in development.**

Planned milestones:

- [ ] Dataset validation and exploratory analysis
- [ ] Data preprocessing pipeline
- [ ] Baseline classification models
- [ ] Model comparison and evaluation
- [ ] Final model selection
- [ ] Explainability module
- [ ] Similar application retrieval
- [ ] Counterfactual analysis engine
- [ ] Prediction reliability analysis
- [ ] Backend API
- [ ] Frontend integration
- [ ] End-to-end testing
- [ ] Final documentation

---

## ⚠️ Disclaimer

LoanLens is an **educational machine learning project**. Its predictions, explanations and counterfactual scenarios describe the behavior of the trained model and should not be interpreted as actual financial advice, lending decisions, or HDFC Bank policy.

The project is designed to demonstrate how machine learning can be made more **interpretable, interactive and decision-aware** in a credit-analysis setting.

---

## 👨‍💻 Project

Built as part of the **AI for Engineers** course at Thapar Institute of Engineering & Technology.

**Repository:** https://github.com/utkarsh-goel18/loanlens
