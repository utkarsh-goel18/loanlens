import joblib
import pandas as pd
import shap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "loanlens_logistic_pipeline.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "loan_approval.csv"


class SHAPService:

    def __init__(self):

        # Load trained pipeline
        self.pipeline = joblib.load(MODEL_PATH)

        # Extract preprocessing and model
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.model = self.pipeline.named_steps["model"]

        # Load training data for SHAP background
        data = pd.read_csv(DATA_PATH)

        # Remove ID and target
        X = data.drop(
            columns=["Loan_Status", "Loan_ID"]
        )

        # Apply the same preprocessing used during training
        X_transformed = self.preprocessor.transform(X)

        # Representative background sample
        background = shap.sample(
            X_transformed,
            100,
            random_state=42
        )

        # SHAP explainer for Logistic Regression
        self.explainer = shap.LinearExplainer(
            self.model,
            background
        )

        # Original numerical features
        self.numeric_features = [
            "Applicant_Income",
            "Coapplicant_Income",
            "Loan_Amount",
            "Loan_Term",
            "Age"
        ]

        # Original categorical features
        self.categorical_features = [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Employment_Status",
            "Credit_History",
            "Property_Area"
        ]

    def _get_impact_label(self, contribution: float) -> str:

        if contribution > 0:
            return "Supports approval"

        if contribution < 0:
            return "Works against approval"

        return "Neutral"

    def explain(self, application: dict):

        # Convert application to DataFrame
        dataframe = pd.DataFrame([application])

        # Apply trained preprocessing
        transformed_data = self.preprocessor.transform(
            dataframe
        )

        # Calculate SHAP values
        shap_values = self.explainer(
            transformed_data
        )

        values = shap_values.values[0]

        # Get transformed feature names
        feature_names = (
            self.preprocessor
            .get_feature_names_out()
        )

        explanations = []

        for feature, value in zip(
            feature_names,
            values
        ):

            # Remove transformer prefixes
            clean_feature = feature.replace(
                "num__",
                ""
            )

            clean_feature = clean_feature.replace(
                "cat__",
                ""
            )

            # Numerical features
            if clean_feature in self.numeric_features:

                contribution = float(value)

                explanations.append({
                    "feature": clean_feature,
                    "value": application[clean_feature],
                    "contribution": contribution,
                    "direction": (
                        "positive"
                        if contribution > 0
                        else "negative"
                    ),
                    "impact": self._get_impact_label(
                        contribution
                    )
                })

                continue

            # Categorical features
            matched_feature = None
            category = None

            for original_feature in (
                self.categorical_features
            ):

                prefix = original_feature + "_"

                if clean_feature.startswith(prefix):

                    matched_feature = original_feature

                    category = clean_feature[
                        len(prefix):
                    ]

                    break

            if matched_feature is not None:

                application_value = str(
                    application[matched_feature]
                )

                # Only report the category that
                # actually applies to the applicant.
                if category == application_value:

                    contribution = float(value)

                    explanations.append({
                        "feature": matched_feature,
                        "value": application[
                            matched_feature
                        ],
                        "contribution": contribution,
                        "direction": (
                            "positive"
                            if contribution > 0
                            else "negative"
                        ),
                        "impact": self._get_impact_label(
                            contribution
                        )
                    })

        # Strongest effects first
        explanations.sort(
            key=lambda x: abs(
                x["contribution"]
            ),
            reverse=True
        )

        # Return top 10 factors
        return explanations[:10]


shap_service = SHAPService()