import joblib
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "loanlens_logistic_pipeline.joblib"


class CounterfactualService:

    def __init__(self):
        self.pipeline = joblib.load(MODEL_PATH)

        # Features that LoanLens is allowed to modify.
        # Credit_History is intentionally excluded because
        # it is not treated as an actionable variable.
        self.actionable_features = [
            "Applicant_Income",
            "Coapplicant_Income",
            "Loan_Amount",
            "Loan_Term"
        ]

        # Loan terms that are commonly represented in the dataset.
        # Loan_Term is treated as a discrete variable rather than
        # an arbitrary continuous number.
        self.valid_loan_terms = [
            120,
            180,
            240,
            300,
            360,
            420,
            480
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

    def _find_numeric_boundary(
        self,
        application: dict,
        feature: str,
        original_prediction: str
    ):

        original_value = float(application[feature])

        successful_candidates = []

        # Search upward and downward.
        for direction in [1, -1]:

            if direction == 1:

                low = original_value

                high = max(
                    original_value * 2,
                    original_value + 1000
                )

                # Expand the search range until the prediction flips.
                for _ in range(10):

                    test_application = application.copy()
                    test_application[feature] = high

                    prediction, _ = self._predict(
                        test_application
                    )

                    if prediction != original_prediction:
                        break

                    high *= 2

                else:
                    continue

            else:

                high = original_value
                low = max(0, original_value * 0.1)

                # Check the lower boundary.
                for _ in range(10):

                    test_application = application.copy()
                    test_application[feature] = low

                    prediction, _ = self._predict(
                        test_application
                    )

                    if prediction != original_prediction:
                        break

                    if low == 0:
                        break

                    low *= 0.5

                else:
                    continue

                # If even zero does not flip the prediction,
                # this direction has no successful counterfactual.
                test_application = application.copy()
                test_application[feature] = low

                prediction, _ = self._predict(
                    test_application
                )

                if prediction == original_prediction:
                    continue

            # Binary search for the smallest change that flips
            # the model prediction.
            for _ in range(30):

                midpoint = (low + high) / 2

                test_application = application.copy()
                test_application[feature] = midpoint

                prediction, _ = self._predict(
                    test_application
                )

                if direction == 1:

                    if prediction != original_prediction:
                        high = midpoint
                    else:
                        low = midpoint

                else:

                    if prediction != original_prediction:
                        low = midpoint
                    else:
                        high = midpoint

            new_value = high if direction == 1 else low

            final_application = application.copy()
            final_application[feature] = new_value

            final_prediction, final_probability = self._predict(
                final_application
            )

            if final_prediction == original_prediction:
                continue

            original_probability = self._predict(
                application
            )[1]

            successful_candidates.append({
                "changed_feature": feature,
                "original_value": original_value,
                "new_value": float(new_value),
                "original_prediction": original_prediction,
                "new_prediction": final_prediction,
                "original_probability": original_probability,
                "new_probability": final_probability,
                "probability_gain": (
                    final_probability - original_probability
                ),
                "change_cost": self._change_cost(
                    original_value,
                    new_value
                )
            })

        if not successful_candidates:
            return None

        successful_candidates.sort(
            key=lambda x: x["change_cost"]
        )

        return successful_candidates[0]

    def _find_loan_term_counterfactual(
        self,
        application: dict,
        original_prediction: str
    ):

        original_value = float(application["Loan_Term"])

        original_probability = self._predict(
            application
        )[1]

        candidates = []

        for loan_term in self.valid_loan_terms:

            # Don't test the value the applicant already has.
            if loan_term == original_value:
                continue

            test_application = application.copy()
            test_application["Loan_Term"] = loan_term

            prediction, probability = self._predict(
                test_application
            )

            if prediction != original_prediction:

                candidates.append({
                    "changed_feature": "Loan_Term",
                    "original_value": original_value,
                    "new_value": float(loan_term),
                    "original_prediction": original_prediction,
                    "new_prediction": prediction,
                    "original_probability": original_probability,
                    "new_probability": probability,
                    "probability_gain": (
                        probability - original_probability
                    ),
                    "change_cost": self._change_cost(
                        original_value,
                        loan_term
                    )
                })

        if not candidates:
            return None

        candidates.sort(
            key=lambda x: x["change_cost"]
        )

        return candidates[0]

    def find_counterfactuals(self, application: dict):

        original_prediction, original_probability = self._predict(
            application
        )

        # If already approved, there is no need to search for
        # changes that make an approval happen.
        if original_prediction == "Approved":

            return {
                "original_prediction": original_prediction,
                "original_probability": original_probability,
                "counterfactuals": [],
                "found": False
            }

        successful_counterfactuals = []

        # Continuous numerical features.
        numeric_features = [
            "Applicant_Income",
            "Coapplicant_Income",
            "Loan_Amount"
        ]

        for feature in numeric_features:

            result = self._find_numeric_boundary(
                application,
                feature,
                original_prediction
            )

            if result is not None:
                successful_counterfactuals.append(result)

        # Discrete Loan_Term search.
        loan_term_result = self._find_loan_term_counterfactual(
            application,
            original_prediction
        )

        if loan_term_result is not None:
            successful_counterfactuals.append(
                loan_term_result
            )

        # Smallest relative change first.
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