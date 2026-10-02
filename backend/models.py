"""
models.py
---------
SQLAlchemy ORM models mapping 1:1 to the tables in sql/schema.sql
"""
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, JSON
)
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    videos = relationship("Video", back_populates="owner", cascade="all, delete-orphan")


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    filepath = Column(String(500), nullable=False)
    processed_path = Column(String(500), nullable=True)
    status = Column(String(20), default="uploaded")  # uploaded, processing, done, failed
    duration_seconds = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    total_frames = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="videos")
    predictions = relationship("Prediction", back_populates="video", cascade="all, delete-orphan")
    detected_objects = relationship("DetectedObject", back_populates="video", cascade="all, delete-orphan")
    activities = relationship("Activity", back_populates="video", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="video", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    model_used = Column(String(50))  # cnn, cnn_lstm, cnn_gru, cnn_attention
    predicted_label = Column(String(100))
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("Video", back_populates="predictions")


class DetectedObject(Base):
    __tablename__ = "detected_objects"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    frame_number = Column(Integer)
    timestamp = Column(Float)
    object_name = Column(String(100))
    confidence = Column(Float)
    bbox_x1 = Column(Float)
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)

    video = relationship("Video", back_populates="detected_objects")


class Activity(Base):
    __tablename__ = "activities"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    activity_label = Column(String(100))
    start_time = Column(Float)
    end_time = Column(Float)
    confidence = Column(Float)

    video = relationship("Video", back_populates="activities")


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    event_type = Column(String(100))
    description = Column(Text)
    timestamp = Column(Float)
    severity = Column(String(20), default="info")  # info, warning, critical

    video = relationship("Video", back_populates="events")


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(50))
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc = Column(Float, nullable=True)
    trained_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)
