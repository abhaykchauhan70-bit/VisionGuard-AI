"""
services/lstm_service.py
---------------------------
Calls ai/inference/predict.py to run the trained CNN-LSTM hybrid on an
uploaded video and returns a structured result the API can serialize.
Never fabricates a result: if no model has been trained yet, it returns
a clear "model_not_trained" status instead of a fake label.
"""
import os
from ai.inference.predict import predict_activity

CHECKPOINT_PATH = os.path.join("models", "cnn_lstm_best.pt")


def run_activity_prediction(video_path: str):
    if not os.path.exists(CHECKPOINT_PATH):
        return {
            "status": "model_not_trained",
            "message": (
                "No trained activity-recognition model found yet. "
                "Train one with: python -m ai.training.train_hybrid "
                "(see README for dataset setup)."
            ),
        }
    try:
        result = predict_activity(video_path, checkpoint_path=CHECKPOINT_PATH)
        result["status"] = "ok"
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}
