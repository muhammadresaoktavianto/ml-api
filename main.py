import os
import joblib
import gdown
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

MODEL_PATH = "rf_model_lead.pkl"
MODEL_URL = "https://drive.google.com/uc?id=1F0O0eQi8rNnICfQXm5UIdHtYJ0PiDMXv"

# Download model jika belum ada
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = joblib.load(MODEL_PATH)

# ==============================
# Input Schema hanya fitur model
# ==============================
class LeadInput(BaseModel):
    age: float
    duration: float
    campaign: float
    pdays: float
    previous: float
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float
    job: str
    marital: str
    education: str
    default_status: str
    housing: str
    loan: str
    contact: str
    month: str
    poutcome: str

# ==============================
# Urutan fitur sesuai model
# ==============================
FEATURE_ORDER = [
    "age", "duration", "campaign", "pdays", "previous",
    "emp_var_rate", "cons_price_idx", "cons_conf_idx",
    "euribor3m", "nr_employed", "job", "marital", "education",
    "default_status", "housing", "loan", "contact", "month",
    "poutcome"
]

@app.get("/")
def home():
    return {"message": "ML API ready 🚀"}

# ==============================
# Prediction Endpoint
# ==============================
@app.post("/predict")
def predict(data: LeadInput):
    d = data.dict()

    # Ambil hanya fitur model
    X = np.array([d[col] for col in FEATURE_ORDER]).reshape(1, -1)

    # Predict
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]  # probabilitas positif

    return {
        "prediction": int(pred),
        "probability": float(prob)
    }
