"""
routes/dashboard.py
----------------------
Aggregate stats + model performance metrics for the dashboard charts/cards.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import get_db
from backend import models, schemas
from backend.auth_utils import get_current_user

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/stats", response_model=schemas.DashboardStats)
def get_stats(db: Session = Depends(get_db),
              current_user: models.User = Depends(get_current_user)):
    video_ids = [v.id for v in db.query(models.Video.id).filter(
        models.Video.user_id == current_user.id
    ).all()]

    total_videos = len(video_ids)
    total_predictions = db.query(models.Prediction).filter(
        models.Prediction.video_id.in_(video_ids)
    ).count() if video_ids else 0
    total_events = db.query(models.Event).filter(
        models.Event.video_id.in_(video_ids)
    ).count() if video_ids else 0

    avg_conf = db.query(func.avg(models.Prediction.confidence)).filter(
        models.Prediction.video_id.in_(video_ids)
    ).scalar() if video_ids else 0

    return schemas.DashboardStats(
        total_videos=total_videos,
        total_predictions=total_predictions,
        total_events=total_events,
        average_confidence=round(float(avg_conf or 0), 4),
    )


@router.get("/activity-distribution")
def activity_distribution(db: Session = Depends(get_db),
                           current_user: models.User = Depends(get_current_user)):
    video_ids = [v.id for v in db.query(models.Video.id).filter(
        models.Video.user_id == current_user.id
    ).all()]
    if not video_ids:
        return {}
    rows = db.query(models.Prediction.predicted_label, func.count(models.Prediction.id)) \
        .filter(models.Prediction.video_id.in_(video_ids)) \
        .group_by(models.Prediction.predicted_label).all()
    return {label: count for label, count in rows}


@router.get("/models/metrics")
def model_metrics(db: Session = Depends(get_db),
                   current_user: models.User = Depends(get_current_user)):
    rows = db.query(models.ModelMetric).order_by(models.ModelMetric.trained_at.desc()).all()
    return [
        {
            "model_name": r.model_name, "accuracy": r.accuracy, "precision": r.precision,
            "recall": r.recall, "f1_score": r.f1_score, "auc": r.auc,
            "trained_at": r.trained_at.isoformat(),
        } for r in rows
    ]
