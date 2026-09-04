import joblib
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "loanlens_logistic_pipeline.joblib"


class CounterfactualService:

    def __init__(self):
        self.pipeline = joblib.load(MODEL_PATH)

        # Variables that LoanLens is allowed to change.
        # Credit_History is deliberately excluded.
        self.actionable_features = [
            "Applicant_Income",
            "Coapplicant_Income",
            "Loan_Amount",
            "Loan_Term"
        ]

    def _predict(self, application: dict):

        dataframe = pd.DataFrame([application])

        prediction = self.pipeline.predict(dataframe)[0]

        probabilities = self.pipeline.predict_proba(dataframe)[0]

        classes = list(self.pipeline.classes_)

        approved_index = classes.index("Approved")

        approval_probability = probabilities[approved_index]

        return prediction, float(approval_probability)

    def _change_cost(self, original, new):

        if original == 0:
            return abs(new - original)

        return abs(new - original) / abs(original)

    def _find_boundary_for_feature(
        self,
        application: dict,
        feature: str,
        original_prediction: str
    ):

        original_value = float(application[feature])

        # We search both directions.
        # For income variables, increasing may help.
        # For loan amount, decreasing may help.
        # The model itself determines whether a direction works.

        directions = [1, -1]

        successful_candidates = []

        for direction in directions:

            if direction == 1:

                low = original_value
                high = max(
                    original_value * 2,
                    original_value + 1000
                )

            else:

                low = max(
                    0,
                    original_value * 0.1
                )
                high = original_value

            # Check whether this direction can actually
            # produce an approved prediction.
            test_application = application.copy()
            test_application[feature] = high

            prediction, probability = self._predict(
                test_application
            )

            if prediction == original_prediction:
                continue

            # Binary search for the smallest change that flips
            # the prediction.
            for _ in range(25):

                midpoint = (low + high) / 2

                test_application = application.copy()
                test_application[feature] = midpoint

                prediction, probability = self._predict(
                    test_application
                )

                if prediction != original_prediction:

                    if direction == 1:
                        high = midpoint
                    else:
                        low = midpoint

                else:

                    if direction == 1:
                        low = midpoint
                    else:
                        high = midpoint

            # The successful boundary value
            if direction == 1:
                new_value = high
            else:
                new_value = low

            final_application = application.copy()
            final_application[feature] = new_value

            final_prediction, final_probability = self._predict(
                final_application
            )

            if final_prediction != original_prediction:

                successful_candidates.append({
                    "changed_feature": feature,
                    "original_value": original_value,
                    "new_value": float(new_value),
                    "original_prediction": original_prediction,
                    "new_prediction": final_prediction,
                    "original_probability": self._predict(
                        application
                    )[1],
                    "new_probability": final_probability,
                    "probability_gain": (
                        final_probability
                        - self._predict(application)[1]
                    ),
                    "change_cost": self._change_cost(
                        original_value,
                        new_value
                    ),
                    "relative_change": (
                        (new_value - original_value)
                        / original_value
                        if original_value != 0
                        else None
                    )
                })

        if not successful_candidates:
            return None

        # For this feature, keep the smallest relative change.
        successful_candidates.sort(
            key=lambda x: x["change_cost"]
        )

        return successful_candidates[0]

    def find_counterfactuals(self, application: dict):

        original_prediction, original_probability = self._predict(
            application
        )

        successful_counterfactuals = []

        # If already approved, there is no need to find a
        # rejection counterfactual.
        if original_prediction == "Approved":

            return {
                "original_prediction": original_prediction,
                "original_probability": original_probability,
                "counterfactuals": [],
                "found": False
            }

        # Search each actionable feature independently.
        for feature in self.actionable_features:

            result = self._find_boundary_for_feature(
                application,
                feature,
                original_prediction
            )

            if result is not None:
                successful_counterfactuals.append(result)

        # Rank by smallest realistic change first.
        successful_counterfactuals.sort(
            key=lambda x: x["change_cost"]
        )

        return {
            "original_prediction": original_prediction,
            "original_probability": original_probability,
            "counterfactuals": successful_counterfactuals[:5],
            "found": len(successful_counterfactuals) > 0
        }


counterfactual_service = CounterfactualService()