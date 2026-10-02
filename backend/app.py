"""
backend/app.py
----------------
FastAPI application entrypoint.

Run from the PROJECT ROOT (VisionGuard-AI/) with:
    uvicorn backend.app:app --reload

Then open:
    Swagger docs:  http://127.0.0.1:8000/docs
    Health check:  http://127.0.0.1:8000/health
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database import init_db
from backend.routes import auth, video, prediction, dashboard

app = FastAPI(
    title="VisionGuard AI",
    description="Real-Time Video Intelligence and Human Activity Understanding System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(video.router)
app.include_router(prediction.router)
app.include_router(dashboard.router)

# Serve processed/result videos and uploaded originals statically so the
# frontend <video> tags can play them directly.
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/results", StaticFiles(directory=settings.RESULTS_DIR), name="results")


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "service": "VisionGuard AI backend"}
