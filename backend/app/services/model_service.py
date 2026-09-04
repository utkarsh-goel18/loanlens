from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "loanlens_logistic_pipeline.joblib"

class ModelService:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        self.model = joblib.load(MODEL_PATH)

    def predict(self, application: dict):
        dataframe = pd.DataFrame([application])

        prediction = self.model.predict(dataframe)[0]
        probability = self.model.predict_proba(dataframe)[0]

        classes = list(self.model.classes_)
        approved_index = classes.index("Approved")

        approval_probability = probability[approved_index]

        return {
            "prediction": prediction,
            "approval_probability": float(approval_probability),
        }

model_service = ModelService()