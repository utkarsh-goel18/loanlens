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

        training_data = self.data.drop(
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
                    training_data[feature].quantile(0.10)
                ),
                "upper": float(
                    training_data[feature].quantile(0.90)
                ),
            }

        bounds["Loan_Term"] = {
            "lower": float(
                training_data["Loan_Term"].quantile(0.10)
            ),
            "upper": float(
                training_data["Loan_Term"].quantile(0.90)
            ),
        }

        return bounds

    def _predict(self, application: dict):
        dataframe = pd.DataFrame([application])

        prediction = self.pipeline.predict(dataframe)[0]

        probabilities = self.pipeline.predict_proba(dataframe)[0]

        classes = list(self.pipeline.classes_)

        # Explicitly locate the Approved class so that
        # this does not depend on class ordering.
        approved_index = classes.index("Approved")

        approval_probability = probabilities[approved_index]

        return prediction, float(approval_probability)

    def _change_cost(
        self,
        original: dict,
        candidate: dict,
    ):
        total = 0.0

        for feature in self.actionable_features:

            lower = self.bounds[feature]["lower"]
            upper = self.bounds[feature]["upper"]

            scale = max(
                upper - lower,
                1e-9
            )

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

        values = np.linspace(
            lower,
            upper,
            points
        )

        if feature == "Loan_Term":

            # Round loan terms to 30-month intervals.
            values = np.round(values / 30) * 30

            values = np.array([
                value
                for value in values
                if value in self.valid_loan_terms
            ])

        else:

            # Round monetary values to practical
            # increments of 1000.
            values = np.round(values / 1000) * 1000

        # Always include the original value so that
        # the search space represents the current
        # application as well.
        values = np.append(
            values,
            original_value
        )

        return np.unique(values)

    def _build_candidate(
        self,
        application: dict,
        changes: dict,
    ):
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
    ):
        new_prediction, new_probability = self._predict(
            candidate
        )

        changes = []

        for feature in self.actionable_features:

            original_value = float(
                application[feature]
            )

            new_value = float(
                candidate[feature]
            )

            if not np.isclose(
                original_value,
                new_value
            ):
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
            "new_probability": new_probability,
            "probability_gain": (
                new_probability
                - original_probability
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
        successes = []

        for feature in self.actionable_features:

            original_value = float(
                application[feature]
            )

            for value in self._candidate_values(
                feature,
                original_value,
                points=15,
            ):

                if np.isclose(
                    value,
                    original_value
                ):
                    continue

                candidate = self._build_candidate(
                    application,
                    {feature: value}
                )

                prediction, _ = self._predict(
                    candidate
                )

                if prediction == "Approved":

                    result = self._build_result(
                        application,
                        candidate,
                        original_prediction,
                        original_probability,
                        "single-feature",
                    )

                    successes.append(result)

        # Prefer the smallest change.
        # If change costs are tied, prefer the
        # counterfactual with the highest probability.
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
        # 7^4 = 2401 combinations.
        grids = {
            feature: self._candidate_values(
                feature,
                float(application[feature]),
                points=7,
            )
            for feature in self.actionable_features
        }

        successes = []

        for values in product(
            *[
                grids[feature]
                for feature in self.actionable_features
            ]
        ):

            changes = {
                feature: value
                for feature, value in zip(
                    self.actionable_features,
                    values
                )
            }

            candidate = self._build_candidate(
                application,
                changes
            )

            prediction, probability = self._predict(
                candidate
            )

            if prediction == "Approved":

                result = self._build_result(
                    application,
                    candidate,
                    original_prediction,
                    original_probability,
                    "multi-feature",
                )

                successes.append(result)

        # Prefer the lowest-cost successful
        # multi-feature scenario.
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
    ):
        grids = {
            feature: self._candidate_values(
                feature,
                float(application[feature]),
                points=7,
            )
            for feature in self.actionable_features
        }

        original_prediction, original_probability = (
            self._predict(application)
        )

        best_result = None

        for values in product(
            *[
                grids[feature]
                for feature in self.actionable_features
            ]
        ):

            candidate = self._build_candidate(
                application,
                {
                    feature: value
                    for feature, value in zip(
                        self.actionable_features,
                        values
                    )
                }
            )

            prediction, probability = self._predict(
                candidate
            )

            result = self._build_result(
                application,
                candidate,
                original_prediction,
                original_probability,
                "best-tested",
            )

            # We want the highest approval probability.
            # If tied, prefer the lower change cost.
            if (
                best_result is None
                or probability
                > best_result["new_probability"]
                or (
                    np.isclose(
                        probability,
                        best_result["new_probability"]
                    )
                    and result["change_cost"]
                    < best_result["change_cost"]
                )
            ):
                best_result = result

        return best_result

    def find_counterfactuals(
        self,
        application: dict,
    ):
        original_prediction, original_probability = (
            self._predict(application)
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
                    "This application is already predicted "
                    "as Approved. No actionable change is "
                    "required to reach the approval threshold."
                ),
            }

        # -------------------------------------------------
        # STAGE 1
        # Search for a single actionable feature change.
        # -------------------------------------------------

        single_feature_successes = (
            self._single_feature_search(
                application,
                original_prediction,
                original_probability,
            )
        )

        if single_feature_successes:

            return {
                "original_prediction": original_prediction,
                "original_probability": original_probability,
                "counterfactuals": (
                    single_feature_successes[:5]
                ),
                "found": True,
                "counterfactual_type": "single-feature",
                "best_tested_probability": None,
                "message": None,
            }

        # -------------------------------------------------
        # STAGE 2
        # Search combinations of actionable features.
        # -------------------------------------------------

        multi_feature_successes = (
            self._multi_feature_search(
                application,
                original_prediction,
                original_probability,
            )
        )

        if multi_feature_successes:

            return {
                "original_prediction": original_prediction,
                "original_probability": original_probability,
                "counterfactuals": (
                    multi_feature_successes[:5]
                ),
                "found": True,
                "counterfactual_type": "multi-feature",
                "best_tested_probability": None,
                "message": None,
            }

        # -------------------------------------------------
        # STAGE 3
        # No realistic scenario changed the prediction.
        # Return the best scenario that was actually tested
        # instead of inventing an actionable recommendation.
        # -------------------------------------------------

        best_tested = self._best_tested_scenario(
            application
        )

        return {
            "original_prediction": original_prediction,
            "original_probability": original_probability,
            "counterfactuals": [],
            "found": False,
            "counterfactual_type": None,
            "best_tested_probability": (
                best_tested["new_probability"]
            ),
            "message": (
                "No realistic actionable scenario "
                "changed the model prediction within "
                "the tested ranges."
            ),
        }


counterfactual_service = CounterfactualService()