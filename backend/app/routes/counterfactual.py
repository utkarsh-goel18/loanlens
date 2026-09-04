from fastapi import APIRouter

from backend.app.schemas import (
    LoanApplication,
    CounterfactualResponse
)

from backend.app.services.counterfactual_service import (
    counterfactual_service
)


router = APIRouter(
    prefix="/counterfactual",
    tags=["Counterfactual Analysis"]
)


@router.post(
    "",
    response_model=CounterfactualResponse
)
def counterfactual(application: LoanApplication):

    return counterfactual_service.find_counterfactuals(
        application.model_dump()
    )