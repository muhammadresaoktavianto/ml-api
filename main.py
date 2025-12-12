from fastapi import FastAPI
import pickle
import numpy as np
import os
import download_model

app = FastAPI()

# Pastikan model sudah didownload
if not os.path.exists("model.pkl"):
    download_model

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "ML API Running Successfully!"}

@app.post("/predict")
def predict(data: dict):
    """
    Format request:
    {
      "features": [value1, value2, value3, ...]
    }
    """
    try:
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}
