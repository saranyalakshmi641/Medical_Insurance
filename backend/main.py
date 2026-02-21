from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
with open("insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    smoker: str
    income: float
    region: str
    claims_count: int
    annual_premium: float

@app.post("/predict")
def predict(data: InsuranceInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {"predicted_annual_medical_cost": float(prediction)}