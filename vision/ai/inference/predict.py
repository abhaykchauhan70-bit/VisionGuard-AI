"""
ai/inference/predict.py
--------------------------
Loads a trained checkpoint and runs inference on a new video clip.
Used by backend/services/lstm_service.py during the /videos/analyze flow.
If no checkpoint exists yet, raises a clear error instead of returning a
fabricated prediction.
"""
import os
import torch
from ai.models.cnn_lstm_model import CNNLSTMHybrid
from backend.services.video_service import extract_frames
import numpy as np


def load_hybrid_model(checkpoint_path="models/cnn_lstm_best.pt", device="cpu"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No trained checkpoint found at {checkpoint_path}. "
            "Run ai/training/train_hybrid.py on your dataset first."
        )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = CNNLSTMHybrid(num_classes=len(ckpt["classes"]), use_attention=True)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["classes"], ckpt.get("seq_len", 8)


def predict_activity(video_path: str, checkpoint_path="models/cnn_lstm_best.pt", device="cpu"):
    model, classes, seq_len = load_hybrid_model(checkpoint_path, device)

    frames = []
    for _, _, frame in extract_frames(video_path, sample_every_n=1, resize_to=(224, 224)):
        frames.append(frame)
        if len(frames) >= seq_len:
            break
    while len(frames) < seq_len and len(frames) > 0:
        frames.append(frames[-1])
    if not frames:
        raise ValueError("Could not read any frames from the video.")

    clip = np.stack(frames, axis=0)
    clip = np.transpose(clip, (0, 3, 1, 2))  # (seq_len, 3, H, W)
    clip_tensor = torch.tensor(clip, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, 3, H, W)

    with torch.no_grad():
        logits, attn_weights = model(clip_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs))

    return {
        "predicted_label": classes[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probs": {classes[i]: float(probs[i]) for i in range(len(classes))},
    }
