from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routes.predict import router as predict_router
from backend.app.routes.explain import router as explain_router
from backend.app.routes.counterfactual import (
    router as counterfactual_router
)
from backend.app.routes.analyze import (
    router as analyze_router
)


app = FastAPI(
    title="LoanLens API",
    description="Explainable AI credit decision analysis API",
    version="1.0.0",
)


# Allow the frontend to communicate with the backend.
# During development, allow requests from any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(predict_router)
app.include_router(explain_router)
app.include_router(counterfactual_router)
app.include_router(analyze_router)


@app.get("/")
def root():
    return {
        "message": "LoanLens API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }