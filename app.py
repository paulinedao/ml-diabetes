from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()  # ← must come first

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

imputer = joblib.load("predict/imputer.pkl")
scaler = joblib.load("predict/scaler.pkl")
model = joblib.load("predict/model.pkl")

COLUMNS_TO_IMPUTE = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
ALL_FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

class Patient(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.get("/")  # ← now app exists
def root():
    return FileResponse("index.html")

@app.post("/predict")
def predict_diabetes(patient: Patient):
    import pandas as pd
    
    data = pd.DataFrame([patient.model_dump()])
    data[COLUMNS_TO_IMPUTE] = data[COLUMNS_TO_IMPUTE].replace(0, np.nan)
    data[COLUMNS_TO_IMPUTE] = imputer.transform(data[COLUMNS_TO_IMPUTE])
    data_scaled = scaler.transform(data[ALL_FEATURES])
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]
    
    return {
        "diabetic": bool(prediction),
        "probability": round(float(probability), 3)
    }