import os
import joblib
import gdown
from fastapi import FastAPI

app = FastAPI()

MODEL_PATH = "model.pkl"
MODEL_URL = "https://drive.google.com/uc?id=1F0O0eQi8rNnICfQXm5UIdHtYJ0PiDMXv"  # ID file kamu

# Download model jika belum ada
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "ML API ready"}

@app.post("/predict")
def predict(data: dict):
    features = data["features"]
    pred = model.predict([features])
    return {"prediction": int(pred[0])}
