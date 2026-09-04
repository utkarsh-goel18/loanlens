from fastapi import FastAPI

from backend.app.routes.predict import router as predict_router
from backend.app.routes.explain import router as explain_router
from backend.app.routes.counterfactual import (
    router as counterfactual_router
)


app = FastAPI(
    title="LoanLens API",
    description="Explainable AI credit decision analysis API",
    version="1.0.0",
)


app.include_router(predict_router)
app.include_router(explain_router)
app.include_router(counterfactual_router)


@app.get("/")
def root():
    return {
        "message": "LoanLens API is running"
    }