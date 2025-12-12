from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import gdown
import os

app = FastAPI()

# ======= Download model dari Google Drive =======
FILE_ID = "1F0O0eQi8rNnICfQXm5UIdHtYJ0PiDMXv"
MODEL_PATH = "rf_model_lead.pkl"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print("Downloading model from Google Drive...")
    gdown.download(url, MODEL_PATH, quiet=False)

# ======= Load model =======
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")

# ======= Request schema =======
class LeadRequest(BaseModel):
    age: int
    duration: int
    campaign: int
    pdays: int
    previous: int
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

# ======= Predict endpoint =======
@app.post("/predict")
def predict(lead: LeadRequest):
    try:
        X = pd.DataFrame([{
            "age": lead.age,
            "duration": lead.duration,
            "campaign": lead.campaign,
            "pdays": lead.pdays,
            "previous": lead.previous,
            "emp.var.rate": lead.emp_var_rate,
            "cons.price.idx": lead.cons_price_idx,
            "cons.conf.idx": lead.cons_conf_idx,
            "euribor3m": lead.euribor3m,
            "nr.employed": lead.nr_employed,
            "job": lead.job,
            "marital": lead.marital,
            "education": lead.education,
            "default": lead.default_status,
            "housing": lead.housing,
            "loan": lead.loan,
            "contact": lead.contact,
            "month": lead.month,
            "poutcome": lead.poutcome
        }])

        pred = model.predict(X)[0]
        return {"prediction": int(pred)}
    except Exception as e:
        return {"error": str(e)}
