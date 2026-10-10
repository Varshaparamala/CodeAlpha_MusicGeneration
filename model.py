"""The neural network: an LSTM that predicts the next musical event."""

from torch import nn


class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch, time) of event ids  ->  logits: (batch, time, vocab_size)
        out, hidden = self.lstm(self.embedding(x), hidden)
        return self.fc(self.dropout(out)), hidden
