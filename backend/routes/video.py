"""
routes/video.py
----------------
Upload, list, retrieve, analyze and delete videos.
Every route here requires a valid JWT (get_current_user dependency) so a
user can only ever see/act on their own videos.
"""
import os
import uuid
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session

from backend.database import get_db, SessionLocal
from backend import models, schemas
from backend.auth_utils import get_current_user
from backend.config import settings
from backend.services.video_service import get_video_metadata, draw_boxes_and_save, extract_frames
from backend.services.detection_service import detect_objects_in_frame
from backend.services.lstm_service import run_activity_prediction

router = APIRouter(prefix="/videos", tags=["Videos"])


def _validate_upload(file: UploadFile):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: mp4, avi, mov")


@router.post("/upload", response_model=schemas.VideoOut, status_code=201)
def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    _validate_upload(file)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    safe_name = f"{uuid.uuid4().hex}{os.path.splitext(file.filename)[1].lower()}"
    dest_path = os.path.join(settings.UPLOAD_DIR, safe_name)

    size_mb = 0
    with open(dest_path, "wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size_mb += len(chunk) / (1024 * 1024)
            if size_mb > settings.MAX_UPLOAD_SIZE_MB:
                out.close()
                os.remove(dest_path)
                raise HTTPException(400, f"File exceeds {settings.MAX_UPLOAD_SIZE_MB} MB limit")
            out.write(chunk)

    try:
        meta = get_video_metadata(dest_path)
    except ValueError as e:
        os.remove(dest_path)
        raise HTTPException(400, str(e))

    video = models.Video(
        user_id=current_user.id,
        filename=safe_name,
        original_filename=file.filename,
        filepath=dest_path,
        status="uploaded",
        duration_seconds=meta["duration_seconds"],
        fps=meta["fps"],
        total_frames=meta["total_frames"],
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    return video


@router.get("", response_model=list[schemas.VideoOut])
def list_videos(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return db.query(models.Video).filter(models.Video.user_id == current_user.id)\
        .order_by(models.Video.uploaded_at.desc()).all()


@router.get("/{video_id}", response_model=schemas.VideoOut)
def get_video(video_id: int, db: Session = Depends(get_db),
              current_user: models.User = Depends(get_current_user)):
    video = db.query(models.Video).filter(
        models.Video.id == video_id, models.Video.user_id == current_user.id
    ).first()
    if not video:
        raise HTTPException(404, "Video not found")
    return video


@router.delete("/{video_id}", status_code=204)
def delete_video(video_id: int, db: Session = Depends(get_db),
                  current_user: models.User = Depends(get_current_user)):
    video = db.query(models.Video).filter(
        models.Video.id == video_id, models.Video.user_id == current_user.id
    ).first()
    if not video:
        raise HTTPException(404, "Video not found")
    if os.path.exists(video.filepath):
        os.remove(video.filepath)
    db.delete(video)
    db.commit()
    return None


def _process_video_job(video_id: int):
    """
    Runs YOLO detection frame-by-frame + activity prediction, saves results.
    NOTE: this runs in a FastAPI BackgroundTask, i.e. AFTER the response has
    already been sent - so it opens its OWN database session rather than
    reusing the request-scoped one (which is closed by then).
    """
    db: Session = SessionLocal()
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        db.close()
        return
    try:
        video.status = "processing"
        db.commit()

        detections_by_frame = {}
        for frame_idx, timestamp, frame in extract_frames(video.filepath, sample_every_n=5):
            frame_uint8 = (frame * 255).astype("uint8")
            dets = detect_objects_in_frame(frame_uint8)
            if dets:
                detections_by_frame[frame_idx] = dets
                for d in dets:
                    db.add(models.DetectedObject(
                        video_id=video.id, frame_number=frame_idx, timestamp=timestamp,
                        object_name=d["name"], confidence=d["confidence"],
                        bbox_x1=d["x1"], bbox_y1=d["y1"], bbox_x2=d["x2"], bbox_y2=d["y2"],
                    ))
        db.commit()

        os.makedirs(settings.RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(settings.RESULTS_DIR, f"processed_{video.filename}")
        draw_boxes_and_save(video.filepath, detections_by_frame, out_path)
        video.processed_path = out_path

        activity_result = run_activity_prediction(video.filepath)
        if activity_result.get("status") == "ok":
            db.add(models.Prediction(
                video_id=video.id, model_used="cnn_lstm_attention",
                predicted_label=activity_result["predicted_label"],
                confidence=activity_result["confidence"],
            ))

        video.status = "done"
        db.commit()
    except Exception:
        video.status = "failed"
        db.commit()
    finally:
        db.close()


@router.post("/{video_id}/analyze")
def analyze_video(
    video_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    video = db.query(models.Video).filter(
        models.Video.id == video_id, models.Video.user_id == current_user.id
    ).first()
    if not video:
        raise HTTPException(404, "Video not found")

    background_tasks.add_task(_process_video_job, video_id)
    video.status = "processing"
    db.commit()
    return {"message": "Analysis started", "video_id": video_id, "status": "processing"}
