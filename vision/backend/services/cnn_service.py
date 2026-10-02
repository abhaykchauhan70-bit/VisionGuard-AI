"""
services/cnn_service.py
--------------------------
Thin wrapper the API routes call. Keeps ai/ (framework-agnostic ML code)
decoupled from backend/ (web framework code).
"""
import torch
from ai.models.cnn_model import CNNFeatureExtractor

_extractor = None

def get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = CNNFeatureExtractor(freeze_backbone=True)
        _extractor.eval()
    return _extractor

def extract_feature_vector(frame_chw_normalized):
    """frame: (3, 224, 224) float32 normalized numpy/tensor -> 512-d vector"""
    extractor = get_extractor()
    tensor = torch.as_tensor(frame_chw_normalized, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return extractor(tensor).squeeze(0).numpy()
