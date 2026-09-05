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


@router.post(
    "",
    response_model=AnalysisResponse
)
def analyze(application: LoanApplication):

    application_data = application.model_dump()

    # Prediction
    prediction_result = model_service.predict(
        application_data
    )

    # SHAP explanation
    explanations = shap_service.explain(
        application_data
    )

    # Counterfactual analysis
    counterfactual_result = (
        counterfactual_service.find_counterfactuals(
            application_data
        )
    )

    return {
        "prediction": prediction_result["prediction"],
        "approval_probability": prediction_result[
            "approval_probability"
        ],

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
        )
    }