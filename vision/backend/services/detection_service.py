"""
services/detection_service.py
------------------------------
Wraps YOLO (Ultralytics) for object detection on individual frames or a
whole video. Downloads the pretrained COCO weights automatically on first run.
"""
from ultralytics import YOLO
import os

_model = None


def get_model():
    """Lazy-load YOLO so the server starts fast and only loads on first use."""
    global _model
    if _model is None:
        weights_path = os.path.join("models", "yolov8n.pt")
        # ultralytics will auto-download yolov8n.pt to this path if missing
        _model = YOLO(weights_path if os.path.exists(weights_path) else "yolov8n.pt")
    return _model


def detect_objects_in_frame(frame_bgr, conf_threshold: float = 0.4):
    """
    Runs YOLO on a single frame (numpy array, BGR, uint8).
    Returns a list of dicts: name, confidence, x1, y1, x2, y2
    """
    model = get_model()
    results = model(frame_bgr, verbose=False, conf=conf_threshold)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "name": name,
            "confidence": round(conf, 4),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    return detections
