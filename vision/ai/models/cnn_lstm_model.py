"""
ai/models/cnn_lstm_model.py
-----------------------------
The full HYBRID model, end-to-end in one nn.Module:

    (batch, seq_len, 3, 224, 224)
        -> CNN applied to every frame independently (TimeDistributed style)
        -> (batch, seq_len, 512)
        -> LSTM/GRU over the sequence
        -> optional attention pooling over timesteps
        -> Fully Connected -> Softmax -> activity class

Configurable sequence length: just change `seq_len` when you build your
batches in the dataset/dataloader - the model itself doesn't hardcode it.

Attention (optional): instead of only using the LSTM's last hidden state,
attention learns a weight for EVERY timestep and takes a weighted sum. This
lets the model focus on the most informative frames in the clip (e.g. the
exact moment someone falls) rather than just trusting the last frame.
"""
import torch
import torch.nn as nn
from ai.models.cnn_model import CNNFeatureExtractor


class TemporalAttention(nn.Module):
    """A simple additive attention layer over the LSTM's per-timestep outputs."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_outputs):
        # rnn_outputs: (batch, seq_len, hidden_dim)
        scores = self.attn(rnn_outputs)             # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)       # normalize over time
        context = torch.sum(weights * rnn_outputs, dim=1)  # (batch, hidden_dim)
        return context, weights.squeeze(-1)


class CNNLSTMHybrid(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=2, num_classes=10,
                 cell_type="LSTM", use_attention=True, freeze_cnn=True, dropout=0.3):
        super().__init__()
        self.cnn = CNNFeatureExtractor(freeze_backbone=freeze_cnn)
        rnn_cls = nn.LSTM if cell_type.upper() == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=self.cnn.output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.use_attention = use_attention
        if use_attention:
            self.attention = TemporalAttention(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, 3, 224, 224) raw frames
        """
        batch, seq_len, c, h, w = x.shape
        x = x.view(batch * seq_len, c, h, w)      # flatten for CNN
        features = self.cnn(x)                     # (batch*seq_len, 512)
        features = features.view(batch, seq_len, -1)  # back to (batch, seq_len, 512)

        rnn_out, _ = self.rnn(features)             # (batch, seq_len, hidden_dim)

        if self.use_attention:
            context, attn_weights = self.attention(rnn_out)
        else:
            context = rnn_out[:, -1, :]
            attn_weights = None

        logits = self.classifier(context)
        return logits, attn_weights


if __name__ == "__main__":
    model = CNNLSTMHybrid(num_classes=6, use_attention=True)
    dummy_clip = torch.randn(2, 8, 3, 224, 224)  # batch=2, 8 frames per clip
    logits, attn = model(dummy_clip)
    print("Logits:", logits.shape)          # torch.Size([2, 6])
    print("Attention weights:", attn.shape)  # torch.Size([2, 8])
