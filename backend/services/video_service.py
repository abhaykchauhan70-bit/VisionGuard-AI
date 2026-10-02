"""
services/video_service.py
--------------------------
OpenCV helpers: read a video, pull metadata (fps/frame count/duration),
extract frames at a fixed sampling rate, resize + normalize for the model.
"""
import os
import cv2
import numpy as np


def get_video_metadata(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_seconds": round(duration, 2),
        "width": width,
        "height": height,
    }


def extract_frames(path: str, sample_every_n: int = 5, resize_to=(224, 224)):
    """
    Yields (frame_number, timestamp_seconds, frame_ndarray) for every Nth frame.
    Frame is resized to `resize_to` and normalized to [0, 1] float32.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every_n == 0:
            resized = cv2.resize(frame, resize_to)
            normalized = resized.astype(np.float32) / 255.0
            timestamp = frame_idx / fps
            yield frame_idx, timestamp, normalized
        frame_idx += 1

    cap.release()


def draw_boxes_and_save(path: str, detections_by_frame: dict, output_path: str):
    """
    detections_by_frame: {frame_number: [ {name, confidence, x1,y1,x2,y2}, ... ]}
    Writes a new video with bounding boxes burned in.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for det in detections_by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 2)
            label = f'{det["name"]} {det["confidence"]:.2f}'
            cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 2)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return output_path
