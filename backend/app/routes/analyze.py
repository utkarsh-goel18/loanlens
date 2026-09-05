from fastapi import APIRouter

from backend.app.schemas import (
    LoanApplication,
    AnalysisResponse
)

from backend.app.services.model_service import model_service
from backend.app.services.shap_service import shap_service
from backend.app.services.counterfactual_service import (
    counterfactual_service
)


router = APIRouter(
    prefix="/analyze",
    tags=["Complete Analysis"]
)


def get_decision_strength(probability: float) -> str:

    decision_margin = abs(probability - 0.5)

    if decision_margin >= 0.40:
        return "Very Strong"

    if decision_margin >= 0.20:
        return "Strong"

    if decision_margin >= 0.10:
        return "Moderate"

    return "Borderline"


@router.post(
    "",
    response_model=AnalysisResponse
)
def analyze(application: LoanApplication):

    application_data = application.model_dump()

    prediction_result = model_service.predict(
        application_data
    )

    approval_probability = prediction_result[
        "approval_probability"
    ]

    decision_margin = abs(
        approval_probability - 0.5
    )

    decision_strength = get_decision_strength(
        approval_probability
    )

    explanations = shap_service.explain(
        application_data
    )

    counterfactual_result = (
        counterfactual_service.find_counterfactuals(
            application_data
        )
    )

    return {
        "prediction": prediction_result["prediction"],
        "approval_probability": approval_probability,
        "decision_margin": decision_margin,
        "decision_strength": decision_strength,

        "explanations": explanations,

        "original_prediction": (
            counterfactual_result["original_prediction"]
        ),
        "original_probability": (
            counterfactual_result["original_probability"]
        ),

        "counterfactuals": (
            counterfactual_result["counterfactuals"]
        ),
        "counterfactual_found": (
            counterfactual_result["found"]
        ),
        "counterfactual_type": (
            counterfactual_result["counterfactual_type"]
        ),
        "best_tested_probability": (
            counterfactual_result["best_tested_probability"]
        ),
        "counterfactual_message": (
            counterfactual_result["message"]
        ),
    }