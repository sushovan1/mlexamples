# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 17:22:25 2024

@author: CSU5KOR
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Initialize the FastAPI app
app = FastAPI()

# Define the request model
class LoanRequest(BaseModel):
    person_gender: str
    person_education: str
    person_home_ownership: str
    loan_intent: str
    previous_loan_defaults_on_file: str
    person_age: int
    person_income: int
    person_emp_exp: int
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: int 
    credit_score: int

# Load the sklearn model and label encoder from pickle files
try:
    with open("model/model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("model/encoder.pkl", "rb") as le_file:
        label_encoder = pickle.load(le_file)
except Exception as e:
    raise RuntimeError(f"Error loading model or label encoder: {e}")

@app.post("/predict-loan-eligibility")
async def predict_loan_eligibility(request: LoanRequest):
    try:
        # Encode categorical variables using the label encoder
        person_gender_encoded = label_encoder["person_gender"].transform([request.person_gender])[0]
        person_education_encoded = label_encoder["person_education"].transform([request.person_education])[0]
        person_home_ownership_encoded = label_encoder["person_home_ownership"].transform([request.person_home_ownership])[0]
        loan_intent_encoded = label_encoder["loan_intent"].transform([request.loan_intent])[0]
        previous_loan_defaults_encoded = label_encoder["previous_loan_defaults_on_file"].transform([request.previous_loan_defaults_on_file])[0]

        # Create the feature array
        features = np.array([
            person_gender_encoded,
            person_education_encoded,
            person_home_ownership_encoded,
            loan_intent_encoded,
            previous_loan_defaults_encoded,
            request.person_age,
            request.person_income,
            request.person_emp_exp,
            request.loan_amnt,
            request.loan_int_rate,
            request.loan_percent_income,
            request.cb_person_cred_hist_length,
            request.credit_score
        ]).reshape(1, -1)

        # Predict loan eligibility
        prediction = model.predict(features)

        # Return the result
        eligibility = "Eligible" if prediction[0] == 1 else "Not Eligible"
        return {"loan_eligibility": eligibility}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
