"""Create simple demo MIDI files so you can try the pipeline without downloading music.

These are rule-based pieces (chord progressions plus a wandering melody), NOT real music.
For your final submission, train on real MIDI files (classical, jazz, ...).

Example:  python make_sample_data.py --count 60
"""

import argparse
import os
import random

from midi_utils import events_to_midi_bytes

MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]
MAJOR_PROGRESSIONS = [[0, 4, 5, 3], [0, 5, 3, 4], [0, 3, 4, 0], [1, 4, 0, 0]]
MINOR_PROGRESSIONS = [[0, 5, 2, 6], [0, 3, 4, 0], [0, 6, 5, 4], [0, 3, 6, 2]]
RHYTHMS = [
    [0.5] * 8,
    [1, 0.5, 0.5, 1, 0.5, 0.5],
    [1, 1, 1, 1],
    [0.5, 0.5, 1, 2],
    [1.5, 0.5, 1, 1],
]


def scale_note(scale, index, tonic, base):
    return base + tonic + scale[index % 7] + 12 * (index // 7)


def make_section(rng, scale, progression, tonic):
    """Four bars: a bass note under a melody that moves mostly by step."""
    events = []
    pos = rng.randint(2, 4)  # melody position, as an index into the scale
    for degree in progression:
        bass = scale_note(scale, degree, tonic, 36)
        fifth = scale_note(scale, degree + 4, tonic, 36)
        chord_tones = [c + 7 * o for c in (degree, degree + 2, degree + 4) for o in (0, 1)]
        chord_tones = [c for c in chord_tones if 1 <= c <= 10]
        rhythm = rng.choice(RHYTHMS)
        for i, dur in enumerate(rhythm):
            if i == 0:  # strong beat: land on a chord tone
                pos = min(chord_tones, key=lambda c: abs(c - pos) + rng.random() * 2)
            else:
                pos = max(1, min(11, pos + rng.choice([-2, -1, -1, 0, 1, 1, 2])))
            melody = scale_note(scale, pos, tonic, 60)
            if i == 0:
                pitches = (bass, melody)
            elif i == len(rhythm) // 2:
                pitches = (fifth, melody)
            else:
                pitches = (melody,)
            events.append((tuple(sorted(pitches)), float(dur)))
    return events


def make_piece(rng):
    tonic = rng.randint(0, 11)  # random key, so the key-normalising step has work to do
    minor = rng.random() < 0.4
    scale = MINOR if minor else MAJOR
    progression = rng.choice(MINOR_PROGRESSIONS if minor else MAJOR_PROGRESSIONS)
    a = make_section(rng, scale, progression, tonic)
    b = make_section(rng, scale, progression, tonic)
    return a + a + b + a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--out", default="midi_data")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for i in range(args.count):
        rng = random.Random(1000 + i)
        data = events_to_midi_bytes(make_piece(rng), bpm=rng.choice([80, 100, 120]))
        with open(os.path.join(args.out, f"sample_{i:03d}.mid"), "wb") as f:
            f.write(data)
    print(f"Wrote {args.count} demo MIDI files to '{args.out}'")


if __name__ == "__main__":
    main()
