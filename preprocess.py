"""Turn a folder of MIDI files into training data for the LSTM.

Steps
1. Read each MIDI file into events: (pitches played together, duration in beats)
2. Transpose every piece to C major / A minor (the white keys), so the model learns one common key
3. Build a vocabulary of unique events and convert all pieces to integer sequences

Run `python preprocess.py midi_data` to see a summary of your dataset.
"""

import os
import sys
from collections import Counter

import numpy as np

from midi_utils import read_midi_events

# Pitch classes of the C major scale (the same notes as A minor): the white keys of a piano
WHITE_KEYS = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)


def detect_shift(events):
    """Semitones to move the piece so that most of its notes land on the white keys.

    This puts major pieces in C major and minor pieces in A minor (both use the same notes).
    """
    hist = np.zeros(12)  # how long each pitch class is played
    for pitches, dur in events:
        for p in pitches:
            hist[p % 12] += dur

    # fit[s] = amount of music on white keys after moving every note up by s semitones
    fit = {s: float(np.dot(np.roll(hist, s), WHITE_KEYS)) for s in range(-6, 6)}
    best = max(fit.values())
    ties = [s for s, value in fit.items() if value == best]
    return int(min(ties, key=abs))  # on a tie, move as little as possible


def transpose(events, shift):
    out = []
    for pitches, dur in events:
        moved = tuple(p + shift for p in pitches if 0 <= p + shift <= 127)
        if moved:
            out.append((moved, dur))
    return out


def load_pieces(folder):
    """Read every .mid / .midi file under `folder` and return a list of event lists."""
    paths = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith((".mid", ".midi")):
                paths.append(os.path.join(root, name))
    paths.sort()
    print(f"Found {len(paths)} MIDI files in '{folder}'")

    pieces = []
    for path in paths:
        try:
            events = read_midi_events(path)
        except Exception as error:  # damaged or unusual files are skipped
            print(f"  skipped {os.path.basename(path)}: {error}")
            continue
        if len(events) < 20:
            continue
        pieces.append(transpose(events, detect_shift(events)))
    return pieces


def build_dataset(pieces, min_count=3):
    """Return (vocab, ids). Events seen fewer than `min_count` times are dropped."""
    counts = Counter(event for piece in pieces for event in piece)
    while True:
        vocab = [event for event, c in counts.most_common() if c >= min_count]
        index = {event: i for i, event in enumerate(vocab)}
        ids = [index[e] for piece in pieces for e in piece if e in index]
        if len(ids) >= 2000 or min_count <= 1:  # tiny dataset: keep rarer events too
            return vocab, np.array(ids, dtype=np.int64)
        min_count -= 1


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "midi_data"
    pieces = load_pieces(folder)
    if not pieces:
        raise SystemExit("No usable MIDI files found.")
    vocab, ids = build_dataset(pieces)
    print(f"Usable pieces:   {len(pieces)}")
    print(f"Total events:    {len(ids)}")
    print(f"Unique events:   {len(vocab)}")
    print("First events of piece 1 (pitches, beats):")
    for event in pieces[0][:8]:
        print("  ", event)
