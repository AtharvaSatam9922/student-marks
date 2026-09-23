from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


model = joblib.load("model.pkl")


@app.post("/")
def predict(study_hours: float):

    prediction = model.predict([[study_hours]])

    return {
        "study_hours": study_hours,
        "predicted_marks": prediction[0]
    } 