from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")


@app.get("/")
def home():
    return FileResponse("main.html")


@app.post("/predict")
def predict(study_hours: float):

    input_data = pd.DataFrame({
        "study_hours": [study_hours]
    })

    prediction = model.predict(input_data)

    return {
        "study_hours": study_hours,
        "predicted_marks": float(prediction[0])
    }