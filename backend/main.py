from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pickle
import pandas as pd
import os

app = FastAPI()

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

# ---------------- INPUT ----------------
class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    smoker: str
    income: float
    region: str
    claims_count: int
    annual_premium: float

# ---------------- API ROUTES ----------------
@app.get("/api")
def home():
    return {"message": "API running 🚀"}

@app.post("/predict")
def predict(data: InsuranceInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {"predicted_cost": float(prediction)}

# ---------------- SERVE FRONTEND SAFELY ----------------
FRONTEND_PATH = "frontend/build"

@app.get("/")
def serve_root():
    return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))

@app.get("/{full_path:path}")
def serve_react(full_path: str):
    file_path = os.path.join(FRONTEND_PATH, full_path)

    if os.path.exists(file_path):
        return FileResponse(file_path)

    return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))
