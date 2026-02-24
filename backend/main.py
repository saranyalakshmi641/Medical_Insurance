from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pickle
import pandas as pd
import os

# ---------------- APP INIT ----------------
app = FastAPI(
    title="Medical Insurance API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODEL ----------------
with open("insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- INPUT SCHEMA ----------------
class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    smoker: str
    income: float
    region: str
    claims_count: int
    annual_premium: float

# ---------------- ROOT API ----------------
@app.get("/api")
def home():
    return {"message": "API is running 🚀"}

# ---------------- PREDICTION API ----------------
@app.post("/predict")
def predict(data: InsuranceInput):
    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)[0]
        return {
            "predicted_annual_medical_cost": float(prediction)
        }
    except Exception as e:
        return {"error": str(e)}

# ---------------- SERVE FRONTEND ----------------
# (Make sure your React build folder exists)
if os.path.exists("frontend/build"):
    app.mount(
        "/",
        StaticFiles(directory="frontend/build", html=True),
        name="frontend"
    )
