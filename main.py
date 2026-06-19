import os
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


load_dotenv()

app = FastAPI(
    title="Student Graduation Prediction API",
    description="Dummy API untuk menguji integrasi frontend sebelum model ML tersedia.",
    version="0.1.0",
)

default_origins = (
    "http://localhost:3000,"
    "http://127.0.0.1:3000,"
    "https://gradewise-seven.vercel.app"
)
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", default_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StudentInput(BaseModel):
    study_hours_per_week: float = Field(ge=0, le=168)
    attendance_percentage: float = Field(ge=0, le=100)
    previous_grade: float = Field(ge=0, le=100)
    assignment_average: float = Field(ge=0, le=100)
    internet_access: bool
    family_support: bool


class PredictionResponse(BaseModel):
    prediction: Literal["aman", "berisiko"]
    probability: float = Field(ge=0, le=1)
    is_dummy: bool
    message: str


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "message": "API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentInput) -> PredictionResponse:
    """Menghasilkan prediksi dummy deterministik untuk pengujian frontend."""
    score = (
        min(student.study_hours_per_week / 20, 1) * 0.20
        + (student.attendance_percentage / 100) * 0.30
        + (student.previous_grade / 100) * 0.25
        + (student.assignment_average / 100) * 0.20
        + (0.03 if student.internet_access else 0)
        + (0.02 if student.family_support else 0)
    )

    prediction: Literal["aman", "berisiko"] = (
        "aman" if score >= 0.65 else "berisiko"
    )

    return PredictionResponse(
        prediction=prediction,
        probability=round(score if prediction == "aman" else 1 - score, 2),
        is_dummy=True,
        message="Prediksi sementara untuk pengujian integrasi frontend.",
    )
