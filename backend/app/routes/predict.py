from fastapi import APIRouter

from backend.app.schemas import LoanApplication, PredictionResponse
from backend.app.services.model_service import model_service

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)

@router.post("", response_model=PredictionResponse)
def predict(application: LoanApplication):
    return model_service.predict(
        application.model_dump()
    )