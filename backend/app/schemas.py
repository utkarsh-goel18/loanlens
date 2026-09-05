from typing import Any

from pydantic import BaseModel, Field


class LoanApplication(BaseModel):

    Gender: str
    Married: str
    Dependents: str
    Education: str
    Employment_Status: str

    Applicant_Income: float = Field(
        ge=0,
        description="Applicant income must be non-negative."
    )

    Coapplicant_Income: float = Field(
        ge=0,
        description="Coapplicant income must be non-negative."
    )

    Loan_Amount: float = Field(
        ge=0,
        description="Loan amount must be non-negative."
    )

    Loan_Term: float = Field(
        gt=0,
        description="Loan term must be greater than zero."
    )

    Credit_History: float = Field(
        description="Credit history must be 0 or 1."
    )

    Property_Area: str

    Age: float = Field(
        gt=0,
        le=100,
        description="Age must be between 0 and 100."
    )


class PredictionResponse(BaseModel):

    prediction: str
    approval_probability: float


class FeatureExplanation(BaseModel):

    feature: str
    value: Any
    contribution: float
    direction: str


class ExplanationResponse(BaseModel):

    explanations: list[FeatureExplanation]


class Counterfactual(BaseModel):

    changed_feature: str
    original_value: Any
    new_value: Any
    original_prediction: str
    new_prediction: str
    original_probability: float
    new_probability: float
    probability_gain: float
    change_cost: float


class CounterfactualResponse(BaseModel):

    original_prediction: str
    original_probability: float
    counterfactuals: list[Counterfactual]
    found: bool


class AnalysisResponse(BaseModel):

    prediction: str
    approval_probability: float
    explanations: list[FeatureExplanation]
    original_prediction: str
    original_probability: float
    counterfactuals: list[Counterfactual]
    counterfactual_found: bool