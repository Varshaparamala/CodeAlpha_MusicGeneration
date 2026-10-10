"""Helpers for reading and writing MIDI, and for turning note events into audio.

An "event" is a tuple (pitches, duration):
  pitches  = tuple of MIDI note numbers played together (one note or a chord)
  duration = beats until the next event starts (one of DURATIONS)
"""

import io
import wave

import mido
import numpy as np

TICKS_PER_BEAT = 480
DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
MAX_NOTES_PER_EVENT = 3
GRID = 4  # note starts are snapped to 1/4 of a beat (16th notes)


def nearest_duration(beats):
    return min(DURATIONS, key=lambda d: abs(d - beats))


def read_midi_events(path):
    """Read a MIDI file and return a list of (pitches, duration) events."""
    mid = mido.MidiFile(path, clip=True)
    tpb = mid.ticks_per_beat if mid.ticks_per_beat and mid.ticks_per_beat > 0 else TICKS_PER_BEAT

    onsets = {}  # grid slot -> set of pitches that start there
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            # channel 9 is drums, which we skip
            if msg.type == "note_on" and msg.velocity > 0 and msg.channel != 9:
                slot = round(tick / tpb * GRID)
                onsets.setdefault(slot, set()).add(msg.note)

    slots = sorted(onsets)
    events = []
    for i, slot in enumerate(slots):
        pitches = sorted(onsets[slot])
        if len(pitches) > MAX_NOTES_PER_EVENT:  # keep the lowest note and the highest ones
            pitches = [pitches[0]] + pitches[-(MAX_NOTES_PER_EVENT - 1):]
        gap = (slots[i + 1] - slot) / GRID if i + 1 < len(slots) else 1.0
        events.append((tuple(pitches), nearest_duration(gap)))
    return events


def events_to_midi_bytes(events, bpm=100):
    """Convert events to the bytes of a .mid file (piano, one track)."""
    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    track.append(mido.Message("program_change", program=0, time=0))

    timed = []  # (absolute tick, order, message); note_off sorts before note_on
    beat = 0.0
    for pitches, dur in events:
        start = round(beat * TICKS_PER_BEAT)
        end = round((beat + dur * 0.95) * TICKS_PER_BEAT)
        for p in pitches:
            timed.append((start, 1, mido.Message("note_on", note=int(p), velocity=80, time=0)))
            timed.append((end, 0, mido.Message("note_off", note=int(p), velocity=0, time=0)))
        beat += dur
    timed.sort(key=lambda item: (item[0], item[1]))

    previous = 0
    for tick, _, msg in timed:
        track.append(msg.copy(time=tick - previous))
        previous = tick

    buffer = io.BytesIO()
    mid.save(file=buffer)
    return buffer.getvalue()


def events_to_wav_bytes(events, bpm=100, sample_rate=22050):
    """Render events to a mono 16-bit WAV with a simple piano-like synth (no soundfont needed)."""
    sec_per_beat = 60.0 / bpm
    total_beats = sum(dur for _, dur in events)
    audio = np.zeros(int((total_beats * sec_per_beat + 2.0) * sample_rate))

    start = 0.0
    for pitches, dur in events:
        hold = max(dur * sec_per_beat * 0.95, 0.15)
        n = int((hold + 0.6) * sample_rate)  # note length plus a short release tail
        t = np.arange(n) / sample_rate
        envelope = np.minimum(1.0, t / 0.01) * np.exp(-2.5 * t)
        envelope *= np.exp(-20.0 * np.clip(t - hold, 0.0, None))
        i = int(start * sample_rate)
        for p in pitches:
            freq = 440.0 * 2.0 ** ((p - 69) / 12.0)
            tone = (
                np.sin(2 * np.pi * freq * t)
                + 0.5 * np.sin(4 * np.pi * freq * t)
                + 0.25 * np.sin(6 * np.pi * freq * t)
            )
            audio[i:i + n] += tone * envelope
        start += dur * sec_per_beat

    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.8
    pcm = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def piano_roll_figure(events):
    """Draw the notes as a piano-roll picture (matplotlib figure)."""
    from matplotlib.figure import Figure  # no pyplot, so figures don't pile up in memory

    fig = Figure(figsize=(9, 3))
    ax = fig.subplots()
    beat = 0.0
    for pitches, dur in events:
        for p in pitches:
            ax.broken_barh([(beat, dur * 0.95)], (p - 0.4, 0.8), color="#6c5ce7")
        beat += dur
    ax.set_xlabel("Time (beats)")
    ax.set_ylabel("MIDI note number")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
