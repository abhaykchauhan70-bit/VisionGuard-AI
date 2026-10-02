"""
ai/training/train_hybrid.py
------------------------------
Trains the CNN-LSTM hybrid activity classifier.

IMPORTANT - READ THIS FIRST:
This script will only run meaningfully once you place a real labelled
dataset under ai/datasets/train and ai/datasets/val (see
ai/preprocessing/dataset.py for the expected folder layout). This project
does NOT ship with fabricated accuracy numbers - you must run this
yourself against real data, and whatever the console prints is your real
result, which then gets saved to the model_metrics table.

Usage (from project root, venv activated):
    python -m ai.training.train_hybrid --epochs 20 --seq_len 8 --batch_size 4
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ai.models.cnn_lstm_model import CNNLSTMHybrid
from ai.preprocessing.dataset import ActivityClipDataset


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            logits, _ = model(clips)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return acc, precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
    parser.add_argument("--data_dir", type=str, default="ai/datasets")
    parser.add_argument("--checkpoint_dir", type=str, default="models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    if not os.path.isdir(train_dir) or not os.listdir(train_dir):
        print(f"[STOP] No training data found in {train_dir}.")
        print("Add class subfolders with .mp4 clips before running this script.")
        return

    train_ds = ActivityClipDataset(train_dir, seq_len=args.seq_len)
    val_ds = ActivityClipDataset(val_dir, seq_len=args.seq_len)
    num_classes = len(train_ds.classes)
    print(f"Classes found: {train_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = CNNLSTMHybrid(num_classes=num_classes, use_attention=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_f1 = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * clips.size(0)

        train_loss = running_loss / len(train_ds)
        val_acc, val_prec, val_recall, val_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} - train_loss={train_loss:.4f} "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            ckpt_path = os.path.join(args.checkpoint_dir, "cnn_lstm_best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "classes": train_ds.classes,
                "seq_len": args.seq_len,
            }, ckpt_path)
            print(f"  -> saved new best checkpoint to {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no val_f1 improvement for "
                      f"{args.patience} epochs).")
                break

    print("Training complete. Best val F1:", round(best_f1, 4))


if __name__ == "__main__":
    main()
