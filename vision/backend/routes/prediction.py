"""
routes/prediction.py
----------------------
Read-only endpoints for predictions and detected objects belonging to a video.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend import models, schemas
from backend.auth_utils import get_current_user

router = APIRouter(prefix="/predictions", tags=["Predictions"])


def _owned_video_or_404(video_id, db, user):
    video = db.query(models.Video).filter(
        models.Video.id == video_id, models.Video.user_id == user.id
    ).first()
    if not video:
        raise HTTPException(404, "Video not found")
    return video


@router.get("/{video_id}", response_model=list[schemas.PredictionOut])
def get_predictions(video_id: int, db: Session = Depends(get_db),
                     current_user: models.User = Depends(get_current_user)):
    _owned_video_or_404(video_id, db, current_user)
    return db.query(models.Prediction).filter(models.Prediction.video_id == video_id).all()


@router.get("/{video_id}/objects", response_model=list[schemas.DetectedObjectOut])
def get_detected_objects(video_id: int, db: Session = Depends(get_db),
                          current_user: models.User = Depends(get_current_user)):
    _owned_video_or_404(video_id, db, current_user)
    return db.query(models.DetectedObject).filter(
        models.DetectedObject.video_id == video_id
    ).order_by(models.DetectedObject.frame_number).all()
