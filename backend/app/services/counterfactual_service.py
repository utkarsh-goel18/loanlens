from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "loanlens_logistic_pipeline.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "loan_approval.csv"


class CounterfactualService:

    def __init__(self):
        self.pipeline = joblib.load(MODEL_PATH)
        self.data = pd.read_csv(DATA_PATH)

        # These are the only features that LoanLens
        # is allowed to modify when generating
        # actionable counterfactuals.
        self.actionable_features = [
            "Applicant_Income",
            "Coapplicant_Income",
            "Loan_Amount",
            "Loan_Term",
        ]

        # Practical loan-term values.
        self.valid_loan_terms = [
            60,
            90,
            120,
            180,
            210,
            240,
            300,
            330,
            360,
        ]

        # Calculate realistic feature ranges from
        # the available dataset.
        self.bounds = self._calculate_bounds()

    def _calculate_bounds(self):
        bounds = {}

        available_data = self.data.drop(
            columns=["Loan_Status", "Loan_ID"]
        )

        # Use the 10th–90th percentile range so that
        # counterfactuals stay within realistic values
        # instead of suggesting extreme changes.
        for feature in [
            "Applicant_Income",
            "Coapplicant_Income",
            "Loan_Amount",
        ]:
            bounds[feature] = {
                "lower": float(
                    available_data[feature].quantile(0.10)
                ),
                "upper": float(
                    available_data[feature].quantile(0.90)
                ),
            }

        bounds["Loan_Term"] = {
            "lower": float(
                available_data["Loan_Term"].quantile(0.10)
            ),
            "upper": float(
                available_data["Loan_Term"].quantile(0.90)
            ),
        }

        return bounds

    def _predict(self, application: dict):
        dataframe = pd.DataFrame([application])

        prediction = self.pipeline.predict(dataframe)[0]
        probabilities = self.pipeline.predict_proba(dataframe)[0]
        classes = list(self.pipeline.classes_)
        approved_index = classes.index("Approved")
        approval_probability = probabilities[approved_index]

        return prediction, float(approval_probability)

    def _predict_batch(self, applications: list[dict]):
        """Predict many candidates in one pipeline call."""
        dataframe = pd.DataFrame(applications)
        predictions = self.pipeline.predict(dataframe)
        probabilities = self.pipeline.predict_proba(dataframe)
        classes = list(self.pipeline.classes_)
        approved_index = classes.index("Approved")

        return predictions, probabilities[:, approved_index].astype(float)

    def _change_cost(self, original: dict, candidate: dict):
        total = 0.0

        for feature in self.actionable_features:
            lower = self.bounds[feature]["lower"]
            upper = self.bounds[feature]["upper"]
            scale = max(upper - lower, 1e-9)

            total += (
                abs(
                    float(candidate[feature])
                    - float(original[feature])
                )
                / scale
            )

        return float(total)

    def _candidate_values(
        self,
        feature: str,
        original_value: float,
        points: int = 15,
    ):
        lower = self.bounds[feature]["lower"]
        upper = self.bounds[feature]["upper"]

        values = np.linspace(lower, upper, points)

        if feature == "Loan_Term":
            values = np.round(values / 30) * 30
            values = np.array([
                value
                for value in values
                if value in self.valid_loan_terms
            ])
        else:
            values = np.round(values / 1000) * 1000

        values = np.append(values, original_value)
        return np.unique(values)

    def _build_candidate(self, application: dict, changes: dict):
        candidate = application.copy()

        for feature, value in changes.items():
            candidate[feature] = float(value)

        return candidate

    def _build_result(
        self,
        application: dict,
        candidate: dict,
        original_prediction: str,
        original_probability: float,
        counterfactual_type: str,
        new_prediction: str | None = None,
        new_probability: float | None = None,
    ):
        # Prediction can be supplied by the batch search so that
        # each successful candidate does not trigger another model call.
        if new_prediction is None or new_probability is None:
            new_prediction, new_probability = self._predict(candidate)

        changes = []

        for feature in self.actionable_features:
            original_value = float(application[feature])
            new_value = float(candidate[feature])

            if not np.isclose(original_value, new_value):
                changes.append({
                    "feature": feature,
                    "from_value": original_value,
                    "to_value": new_value,
                })

        return {
            "counterfactual_type": counterfactual_type,
            "original_prediction": original_prediction,
            "new_prediction": new_prediction,
            "original_probability": original_probability,
            "new_probability": float(new_probability),
            "probability_gain": float(
                new_probability - original_probability
            ),
            "change_cost": self._change_cost(
                application,
                candidate,
            ),
            "changes": changes,
        }

    def _single_feature_search(
        self,
        application: dict,
        original_prediction: str,
        original_probability: float,
    ):
        candidates = []

        for feature in self.actionable_features:
            original_value = float(application[feature])

            for value in self._candidate_values(
                feature,
                original_value,
                points=15,
            ):
                if np.isclose(value, original_value):
                    continue

                candidates.append(
                    self._build_candidate(
                        application,
                        {feature: value},
                    )
                )

        if not candidates:
            return []

        predictions, probabilities = self._predict_batch(candidates)
        successes = []

        for candidate, prediction, probability in zip(
            candidates,
            predictions,
            probabilities,
        ):
            if prediction == "Approved":
                successes.append(
                    self._build_result(
                        application,
                        candidate,
                        original_prediction,
                        original_probability,
                        "single-feature",
                        new_prediction=str(prediction),
                        new_probability=float(probability),
                    )
                )

        successes.sort(
            key=lambda item: (
                item["change_cost"],
                -item["new_probability"],
            )
        )

        return successes

    def _multi_feature_search(
        self,
        application: dict,
        original_prediction: str,
        original_probability: float,
    ):
        # Seven candidate points per actionable feature.
        # 7^4 = 2401 combinations, evaluated in one batch.
        grids = {
            feature: self._candidate_values(
                feature,
                float(application[feature]),
                points=7,
            )
            for feature in self.actionable_features
        }

        candidates = []

        for values in product(
            *[
                grids[feature]
                for feature in self.actionable_features
            ]
        ):
            candidates.append(
                self._build_candidate(
                    application,
                    {
                        feature: value
                        for feature, value in zip(
                            self.actionable_features,
                            values,
                        )
                    },
                )
            )

        predictions, probabilities = self._predict_batch(candidates)
        successes = []

        for candidate, prediction, probability in zip(
            candidates,
            predictions,
            probabilities,
        ):
            if prediction == "Approved":
                successes.append(
                    self._build_result(
                        application,
                        candidate,
                        original_prediction,
                        original_probability,
                        "multi-feature",
                        new_prediction=str(prediction),
                        new_probability=float(probability),
                    )
                )

        successes.sort(
            key=lambda item: (
                item["change_cost"],
                -item["new_probability"],
            )
        )

        return successes

    def _best_tested_scenario(
        self,
        application: dict,
        original_prediction: str,
        original_probability: float,
    ):
        # This uses the same multi-feature grid as Stage 2.
        # Predictions are already evaluated in batch here, so a
        # failed counterfactual search does not require another
        # 2401 individual model calls.
        grids = {
            feature: self._candidate_values(
                feature,
                float(application[feature]),
                points=7,
            )
            for feature in self.actionable_features
        }

        candidates = []

        for values in product(
            *[
                grids[feature]
                for feature in self.actionable_features
            ]
        ):
            candidates.append(
                self._build_candidate(
                    application,
                    {
                        feature: value
                        for feature, value in zip(
                            self.actionable_features,
                            values,
                        )
                    },
                )
            )

        predictions, probabilities = self._predict_batch(candidates)

        best_index = int(np.argmax(probabilities))
        best_candidate = candidates[best_index]

        return self._build_result(
            application,
            best_candidate,
            original_prediction,
            original_probability,
            "best-tested",
            new_prediction=str(predictions[best_index]),
            new_probability=float(probabilities[best_index]),
        )

    def find_counterfactuals(self, application: dict):
        original_prediction, original_probability = self._predict(
            application
        )

        # Already approved applications do not need
        # an approval-flipping counterfactual.
        if original_prediction == "Approved":
            return {
                "original_prediction": original_prediction,
                "original_probability": original_probability,
                "counterfactuals": [],
                "found": False,
                "counterfactual_type": None,
                "best_tested_probability": None,
                "message": (
                    "This application is already predicted as Approved. "
                    "No actionable change is required to reach the "
                    "approval threshold."
                ),
            }

        # Stage 1: single actionable feature changes.
        single_feature_successes = self._single_feature_search(
            application,
            original_prediction,
            original_probability,
        )

        if single_feature_successes:
            return {
                "original_prediction": original_prediction,
                "original_probability": original_probability,
                "counterfactuals": single_feature_successes[:5],
                "found": True,
                "counterfactual_type": "single-feature",
                "best_tested_probability": None,
                "message": None,
            }

        # Stage 2: combinations of actionable features.
        multi_feature_successes = self._multi_feature_search(
            application,
            original_prediction,
            original_probability,
        )

        if multi_feature_successes:
            return {
                "original_prediction": original_prediction,
                "original_probability": original_probability,
                "counterfactuals": multi_feature_successes[:5],
                "found": True,
                "counterfactual_type": "multi-feature",
                "best_tested_probability": None,
                "message": None,
            }

        # Stage 3: no realistic scenario changed the prediction.
        # Return the best scenario actually tested instead of inventing
        # an actionable recommendation.
        best_tested = self._best_tested_scenario(
            application,
            original_prediction,
            original_probability,
        )

        return {
            "original_prediction": original_prediction,
            "original_probability": original_probability,
            "counterfactuals": [],
            "found": False,
            "counterfactual_type": None,
            "best_tested_probability": best_tested["new_probability"],
            "message": (
                "No realistic actionable scenario changed the model "
                "prediction within the tested ranges."
            ),
        }


counterfactual_service = CounterfactualService()
