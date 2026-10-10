"""Generate new music with a trained model and save it as MIDI and WAV.

Example:  python generate.py --length 200 --temperature 0.9
"""

import argparse
import os
import random

import torch

from midi_utils import events_to_midi_bytes, events_to_wav_bytes
from model import MusicLSTM


def load_checkpoint(path="model.pt", device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = MusicLSTM(len(ckpt["vocab"]), **ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["vocab"], ckpt["ids"]


@torch.no_grad()
def generate_events(model, vocab, ids, length=150, temperature=0.9, top_k=20, seed_len=16):
    """Pick a short random seed from the training music, then keep predicting what comes next.

    temperature: low = safe and repetitive, high = surprising and messy
    top_k:       only the k most likely events can be chosen at each step
    """
    device = next(model.parameters()).device
    start = random.randint(0, len(ids) - seed_len - 1)
    seed = ids[start:start + seed_len].tolist()

    logits, hidden = model(torch.tensor([seed], dtype=torch.long, device=device))
    next_logits = logits[0, -1]

    chosen = []
    for _ in range(length):
        scaled = next_logits / max(temperature, 1e-3)
        top_values, top_index = torch.topk(scaled, min(top_k, scaled.numel()))
        probs = torch.softmax(top_values, dim=-1)
        event_id = top_index[torch.multinomial(probs, 1)].item()
        chosen.append(event_id)
        step = torch.tensor([[event_id]], dtype=torch.long, device=device)
        logits, hidden = model(step, hidden)
        next_logits = logits[0, -1]

    return [vocab[i] for i in chosen]


def main():
    parser = argparse.ArgumentParser(description="Generate music with the trained LSTM")
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--length", type=int, default=200, help="number of notes/chords")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--bpm", type=int, default=100)
    parser.add_argument("--out", default="output/generated", help="file name without extension")
    args = parser.parse_args()

    model, vocab, ids = load_checkpoint(args.model)
    events = generate_events(model, vocab, ids, args.length, args.temperature, args.top_k)

    folder = os.path.dirname(args.out)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(args.out + ".mid", "wb") as f:
        f.write(events_to_midi_bytes(events, args.bpm))
    with open(args.out + ".wav", "wb") as f:
        f.write(events_to_wav_bytes(events, args.bpm))
    print(f"Saved {args.out}.mid and {args.out}.wav ({len(events)} events)")


if __name__ == "__main__":
    main()
