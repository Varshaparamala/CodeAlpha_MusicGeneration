"""Train the LSTM on a folder of MIDI files.

Example:  python train.py --data midi_data --epochs 30
"""

import argparse
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import MusicLSTM
from preprocess import build_dataset, load_pieces

MODEL_CONFIG = {"embed_dim": 128, "hidden_dim": 256, "num_layers": 2, "dropout": 0.3}


class WindowDataset(Dataset):
    """Sliding windows over the event sequence. Target = input shifted by one step."""

    def __init__(self, ids, seq_len, stride):
        self.ids = torch.as_tensor(ids, dtype=torch.long)
        self.seq_len = seq_len
        self.starts = list(range(0, len(ids) - seq_len, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        chunk = self.ids[s:s + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total += loss.item() * x.size(0)
        count += x.size(0)
    return total / count


def save_checkpoint(path, model, vocab, ids):
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": [(tuple(int(p) for p in pitches), float(d)) for pitches, d in vocab],
            "ids": torch.as_tensor(ids, dtype=torch.long),
            "config": MODEL_CONFIG,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train the music generation LSTM")
    parser.add_argument("--data", default="midi_data", help="folder with .mid files")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=48, help="events per training window")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--min-count", type=int, default=3, help="drop events seen fewer times")
    parser.add_argument("--out", default="model.pt")
    args = parser.parse_args()

    pieces = load_pieces(args.data)
    if not pieces:
        raise SystemExit(
            "No usable MIDI files found. Put .mid files in the folder, "
            "or run `python make_sample_data.py` to create demo files."
        )
    vocab, ids = build_dataset(pieces, args.min_count)
    print(f"{len(pieces)} pieces, {len(ids)} events, {len(vocab)} unique events")
    if len(ids) < args.seq_len * 10:
        raise SystemExit("Not enough music to train on. Add more MIDI files.")

    split = int(len(ids) * 0.9)  # last 10% of the music is used to check for overfitting
    train_ds = WindowDataset(ids[:split], args.seq_len, max(1, args.seq_len // 2))
    val_ds = WindowDataset(ids[split:], args.seq_len, args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    use_val = len(val_ds) > 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}: {len(train_ds)} windows per epoch")
    model = MusicLSTM(len(vocab), **MODEL_CONFIG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        total, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * x.size(0)
            count += x.size(0)

        train_loss = total / count
        val_loss = evaluate(model, val_loader, loss_fn, device) if use_val else train_loss
        note = ""
        if val_loss < best:
            best = val_loss
            save_checkpoint(args.out, model, vocab, ids)
            note = "  <- saved"
        print(
            f"Epoch {epoch:3d}/{args.epochs}  train loss {train_loss:.3f}  "
            f"val loss {val_loss:.3f}  {time.time() - started:.0f}s{note}"
        )

    print(f"Done. Best model saved to {args.out}")


if __name__ == "__main__":
    main()
