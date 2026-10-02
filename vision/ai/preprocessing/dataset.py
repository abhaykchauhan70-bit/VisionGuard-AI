"""
ai/preprocessing/dataset.py
-----------------------------
PyTorch Dataset for activity-recognition clips.

Expected folder layout for YOUR dataset (you provide this - see README):

    ai/datasets/
        train/
            walking/clip_001.mp4, clip_002.mp4, ...
            running/clip_001.mp4, ...
            falling/...
        val/
            walking/...
            running/...
        test/
            walking/...
            running/...

Each subfolder name under train/val/test IS the class label.
Public datasets that fit this task: UCF101, HMDB51, Kinetics-400 (subset),
or your own recorded clips for a small custom activity set.
"""
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from backend.services.video_service import extract_frames


class ActivityClipDataset(Dataset):
    def __init__(self, root_dir: str, seq_len: int = 8, transform=None):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform

        self.classes = sorted(
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".mp4", ".avi", ".mov")):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def _sample_clip(self, video_path):
        frames = []
        for _, _, frame in extract_frames(video_path, sample_every_n=1, resize_to=(224, 224)):
            frames.append(frame)
            if len(frames) >= self.seq_len:
                break
        # pad with the last frame if the clip is shorter than seq_len
        while len(frames) < self.seq_len and len(frames) > 0:
            frames.append(frames[-1])
        clip = np.stack(frames, axis=0)               # (seq_len, H, W, 3)
        clip = np.transpose(clip, (0, 3, 1, 2))         # (seq_len, 3, H, W)
        return torch.tensor(clip, dtype=torch.float32)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        clip = self._sample_clip(path)
        if self.transform:
            clip = self.transform(clip)
        return clip, label
