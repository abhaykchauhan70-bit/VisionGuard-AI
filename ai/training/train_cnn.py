"""
ai/training/train_cnn.py
---------------------------
Fine-tunes the CNN alone as a single-frame classifier (baseline model,
used in the model-comparison experiment against CNN+LSTM/GRU/Attention).
Unfreezes the backbone and trains a classification head on top.

Usage:
    python -m ai.training.train_cnn --epochs 15
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ai.models.cnn_model import CNNFeatureExtractor
from ai.preprocessing.dataset import ActivityClipDataset


class CNNBaseline(nn.Module):
    """CNN-only baseline: classifies using just the middle frame of each clip."""
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = CNNFeatureExtractor(freeze_backbone=False)
        self.classifier = nn.Linear(self.cnn.output_dim, num_classes)

    def forward(self, single_frame_batch):
        feats = self.cnn(single_frame_batch)
        return self.classifier(feats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_dir", type=str, default="ai/datasets")
    parser.add_argument("--checkpoint_dir", type=str, default="models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = os.path.join(args.data_dir, "train")
    if not os.path.isdir(train_dir) or not os.listdir(train_dir):
        print(f"[STOP] No training data found in {train_dir}. Add labelled clips first.")
        return

    train_ds = ActivityClipDataset(train_dir, seq_len=1)  # 1 frame per "clip" for this baseline
    num_classes = len(train_ds.classes)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = CNNBaseline(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        all_preds, all_labels, running_loss = [], [], 0.0
        for clips, labels in train_loader:
            frame = clips[:, 0, :, :, :].to(device)  # take the single frame
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(frame)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * frame.size(0)
            all_preds.extend(torch.argmax(logits, 1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch}/{args.epochs} - loss={running_loss/len(train_ds):.4f} acc={acc:.4f}")

    torch.save({"model_state": model.state_dict(), "classes": train_ds.classes},
               os.path.join(args.checkpoint_dir, "cnn_baseline.pt"))
    print("Saved models/cnn_baseline.pt")


if __name__ == "__main__":
    main()
