import os
import joblib
import gdown
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

MODEL_PATH = "model.pkl"
MODEL_URL = "https://drive.google.com/uc?id=1F0O0eQi8rNnICfQXm5UIdHtYJ0PiDMXv"

# Download model jika belum ada
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = joblib.load(MODEL_PATH)

# ==============================
#   Input Schema (sesuai React)
# ==============================
class LeadInput(BaseModel):
    name: str
    gender: str
    phone_number: str
    age: str
    job: str
    marital: str
    education: str
    default_status: str
    housing: str
    loan: str
    contact: str
    month: str
    day: str
    duration: str
    campaign: str
    pdays: str
    previous: str
    poutcome: str

    emp_var_rate: str
    cons_price_idx: str
    cons_conf_idx: str
    euribor3m: str
    nr_employed: str

    lead_score: str
    status_kampanye: str
    aktivitas: str
    alasan_status: str
    subscription_status: str


# ==============================
#   Urutan fitur sesuai model
# ==============================
FEATURE_ORDER = [
    "name", "gender", "phone_number", "age", "job", "marital", "education",
    "default_status", "housing", "loan", "contact", "month", "day",
    "duration", "campaign", "pdays", "previous", "poutcome",
    "emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m",
    "nr_employed", "lead_score", "status_kampanye", "aktivitas",
    "alasan_status", "subscription_status"
]


@app.get("/")
def home():
    return {"message": "ML API ready 🚀"}


# ==============================
#   Prediction Endpoint
# ==============================
@app.post("/predict")
def predict(data: LeadInput):

    # Convert input → dict
    d = data.dict()

    # Convert ke array sesuai urutan kolom model
    X = np.array([d[col] for col in FEATURE_ORDER]).reshape(1, -1)

    # Predict
    pred = model.predict(X)[0]

    return {
        "prediction": int(pred)
    }
