import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Seluruh artefak dimuat satu kali ketika aplikasi mulai.
model = joblib.load(BASE_DIR / "model_student_performance.pkl")
label_encoders = joblib.load(BASE_DIR / "label_encoders.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
feature_names: list[str] = joblib.load(BASE_DIR / "feature_names.pkl")

app = FastAPI(
    title="Student Performance Prediction API",
    description="API untuk memprediksi nilai akademik mahasiswa.",
    version="1.0.0",
)

default_origins = (
    "http://localhost:3000,"
    "http://localhost:3001,"
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


Level = Literal["Low", "Medium", "High"]
YesNo = Literal["No", "Yes"]


class StudentInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "hours_studied": 20,
                "attendance": 85,
                "parental_involvement": "Medium",
                "access_to_resources": "High",
                "extracurricular_activities": "Yes",
                "sleep_hours": 7,
                "previous_scores": 75,
                "motivation_level": "Medium",
                "internet_access": "Yes",
                "tutoring_sessions": 2,
                "family_income": "Medium",
                "teacher_quality": "High",
                "school_type": "Public",
                "peer_influence": "Positive",
                "physical_activity": 3,
                "learning_disabilities": "No",
                "parental_education_level": "College",
                "distance_from_home": "Near",
                "gender": "Female",
            }
        }
    )

    hours_studied: float = Field(ge=0, le=168)
    attendance: float = Field(ge=0, le=100)
    parental_involvement: Level
    access_to_resources: Level
    extracurricular_activities: YesNo
    sleep_hours: float = Field(ge=0, le=24)
    previous_scores: float = Field(ge=0, le=100)
    motivation_level: Level
    internet_access: YesNo
    tutoring_sessions: int = Field(ge=0)
    family_income: Level
    teacher_quality: Level
    school_type: Literal["Private", "Public"]
    peer_influence: Literal["Negative", "Neutral", "Positive"]
    physical_activity: float = Field(ge=0, le=24)
    learning_disabilities: YesNo
    parental_education_level: Literal["High School", "College", "Postgraduate"]
    distance_from_home: Literal["Near", "Moderate", "Far"]
    gender: Literal["Female", "Male"]


class PredictionResponse(BaseModel):
    predicted_score: float
    is_dummy: bool
    message: str


@app.get("/health")
def health_check() -> dict[str, object]:
    return {
        "status": "ok",
        "message": "API and prediction model are ready",
        "model_loaded": True,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentInput) -> PredictionResponse:
    input_by_feature = {
        "Hours_Studied": student.hours_studied,
        "Attendance": student.attendance,
        "Parental_Involvement": student.parental_involvement,
        "Access_to_Resources": student.access_to_resources,
        "Extracurricular_Activities": student.extracurricular_activities,
        "Sleep_Hours": student.sleep_hours,
        "Previous_Scores": student.previous_scores,
        "Motivation_Level": student.motivation_level,
        "Internet_Access": student.internet_access,
        "Tutoring_Sessions": student.tutoring_sessions,
        "Family_Income": student.family_income,
        "Teacher_Quality": student.teacher_quality,
        "School_Type": student.school_type,
        "Peer_Influence": student.peer_influence,
        "Physical_Activity": student.physical_activity,
        "Learning_Disabilities": student.learning_disabilities,
        "Parental_Education_Level": student.parental_education_level,
        "Distance_from_Home": student.distance_from_home,
        "Gender": student.gender,
    }

    input_frame = pd.DataFrame(
        [[input_by_feature[name] for name in feature_names]],
        columns=feature_names,
    )

    for column, encoder in label_encoders.items():
        input_frame[column] = encoder.transform(input_frame[column])

    scaled_values = scaler.transform(input_frame)
    scaled_frame = pd.DataFrame(scaled_values, columns=feature_names)
    predicted_score = float(model.predict(scaled_frame)[0])

    return PredictionResponse(
        predicted_score=round(predicted_score, 2),
        is_dummy=False,
        message="Prediksi berhasil dibuat menggunakan model student performance.",
    )
