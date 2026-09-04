from app.services.model_service import model_service


application = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Employment_Status": "Salaried",
    "Applicant_Income": 50000,
    "Coapplicant_Income": 20000,
    "Loan_Amount": 150000,
    "Loan_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban",
    "Age": 30,
}


result = model_service.predict(application)

print(result)