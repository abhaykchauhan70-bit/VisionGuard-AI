"""
ai/models/lstm_model.py
-------------------------
Takes a SEQUENCE of CNN feature vectors (one per sampled frame) and
predicts an activity label for the whole clip.

Pipeline recap:
    video frames -> CNN (per frame) -> sequence of feature vectors
                 -> LSTM/GRU (models how the vectors change over time)
                 -> fully-connected classifier -> softmax -> activity label

Why LSTM/GRU instead of just averaging the CNN features?
  A CNN alone sees each frame independently - it has no notion of "motion"
  or "order". Activities like "sitting down" vs "standing up" look like the
  same two poses in reverse order, so it's really an ORDER problem, not a
  single-image problem. Recurrent layers carry a hidden state across
  timesteps, so they learn temporal patterns.
"""
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2,
                 num_classes=10, cell_type="LSTM", bidirectional=False, dropout=0.3):
        super().__init__()
        rnn_cls = nn.LSTM if cell_type.upper() == "LSTM" else nn.GRU

        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        direction_mult = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * direction_mult, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim) - a sequence of CNN feature vectors
        returns: (batch, num_classes) raw logits (apply softmax outside, or
                 use CrossEntropyLoss which does it internally)
        """
        out, _ = self.rnn(x)         # out: (batch, seq_len, hidden_dim)
        last_step = out[:, -1, :]     # take the final timestep's hidden state
        logits = self.classifier(last_step)
        return logits


if __name__ == "__main__":
    model = LSTMClassifier(num_classes=6)
    dummy_seq = torch.randn(4, 16, 512)  # batch=4, 16 frames, 512-dim features
    out = model(dummy_seq)
    print("Logits shape:", out.shape)  # torch.Size([4, 6])
