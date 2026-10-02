"""
schemas.py
----------
Pydantic models: define the shape of request bodies / API responses.
Keeping these separate from SQLAlchemy models (models.py) is best practice -
it stops your DB structure leaking directly into your public API contract.
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


# ---------- Auth ----------
class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=6)

class UserLogin(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ---------- Video ----------
class VideoOut(BaseModel):
    id: int
    filename: str
    original_filename: str
    status: str
    duration_seconds: Optional[float]
    fps: Optional[float]
    total_frames: Optional[int]
    uploaded_at: datetime
    class Config:
        from_attributes = True


# ---------- Detected objects / predictions ----------
class DetectedObjectOut(BaseModel):
    frame_number: int
    timestamp: float
    object_name: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    class Config:
        from_attributes = True

class PredictionOut(BaseModel):
    model_used: str
    predicted_label: str
    confidence: float
    class Config:
        from_attributes = True

class DashboardStats(BaseModel):
    total_videos: int
    total_predictions: int
    total_events: int
    average_confidence: float
