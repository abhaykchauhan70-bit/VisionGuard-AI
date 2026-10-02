"""
ai/training/train_lstm.py
----------------------------
Trains the LSTM classifier on top of FROZEN, pre-extracted CNN features
(faster than the full hybrid - the CNN backbone doesn't get fine-tuned here).
This is the "CNN + LSTM" entry in the model-comparison table.

Usage:
    python -m ai.training.train_lstm --epochs 20 --cell_type LSTM
    python -m ai.training.train_lstm --epochs 20 --cell_type GRU
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ai.models.cnn_model import CNNFeatureExtractor
from ai.models.lstm_model import LSTMClassifier
from ai.preprocessing.dataset import ActivityClipDataset


def extract_sequence_features(cnn, clips, device):
    # clips: (batch, seq_len, 3, H, W) -> features: (batch, seq_len, 512)
    batch, seq_len, c, h, w = clips.shape
    flat = clips.view(batch * seq_len, c, h, w).to(device)
    with torch.no_grad():
        feats = cnn(flat)
    return feats.view(batch, seq_len, -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cell_type", type=str, default="LSTM", choices=["LSTM", "GRU"])
    parser.add_argument("--data_dir", type=str, default="ai/datasets")
    parser.add_argument("--checkpoint_dir", type=str, default="models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = os.path.join(args.data_dir, "train")
    if not os.path.isdir(train_dir) or not os.listdir(train_dir):
        print(f"[STOP] No training data found in {train_dir}. Add labelled clips first.")
        return

    train_ds = ActivityClipDataset(train_dir, seq_len=args.seq_len)
    num_classes = len(train_ds.classes)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    cnn = CNNFeatureExtractor(freeze_backbone=True).to(device).eval()
    model = LSTMClassifier(num_classes=num_classes, cell_type=args.cell_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        all_preds, all_labels, running_loss = [], [], 0.0
        for clips, labels in train_loader:
            labels = labels.to(device)
            seq_feats = extract_sequence_features(cnn, clips, device)
            optimizer.zero_grad()
            logits = model(seq_feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * clips.size(0)
            all_preds.extend(torch.argmax(logits, 1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch}/{args.epochs} - loss={running_loss/len(train_ds):.4f} acc={acc:.4f}")

    fname = f"cnn_{args.cell_type.lower()}.pt"
    torch.save({"model_state": model.state_dict(), "classes": train_ds.classes,
                "cell_type": args.cell_type}, os.path.join(args.checkpoint_dir, fname))
    print(f"Saved models/{fname}")


if __name__ == "__main__":
    main()
