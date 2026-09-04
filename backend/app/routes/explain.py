from fastapi import APIRouter

from backend.app.schemas import (
    LoanApplication,
    ExplanationResponse
)

from backend.app.services.shap_service import shap_service


router = APIRouter(
    prefix="/explain",
    tags=["Explainability"]
)


@router.post("", response_model=ExplanationResponse)
def explain(application: LoanApplication):

    explanations = shap_service.explain(
        application.model_dump()
    )

    return {
        "explanations": explanations
    }