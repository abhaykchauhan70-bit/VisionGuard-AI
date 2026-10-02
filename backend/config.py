"""
config.py
---------
Central configuration loaded from environment variables (.env file).
Never hardcode secrets here - everything sensitive comes from .env
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "3306")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_NAME = os.getenv("DB_NAME", "visionguard")

    DATABASE_URL = (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-in-production")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
    RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
    MODELS_DIR = os.getenv("MODELS_DIR", "models")
    MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))
    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}

    FRONTEND_ORIGINS = os.getenv(
        "FRONTEND_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500"
    ).split(",")

settings = Settings()
