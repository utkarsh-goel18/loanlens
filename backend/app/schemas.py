from pydantic import BaseModel


class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Employment_Status: str
    Applicant_Income: float
    Coapplicant_Income: float
    Loan_Amount: float
    Loan_Term: float
    Credit_History: float
    Property_Area: str
    Age: float


class PredictionResponse(BaseModel):
    prediction: str
    approval_probability: float


class FeatureExplanation(BaseModel):
    feature: str
    value: object
    contribution: float
    direction: str


class ExplanationResponse(BaseModel):
    explanations: list[FeatureExplanation]


class Counterfactual(BaseModel):
    changed_feature: str
    original_value: object
    new_value: object
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