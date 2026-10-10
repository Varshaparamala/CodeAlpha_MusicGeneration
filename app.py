import os

import streamlit as st

from generate import generate_events, load_checkpoint
from midi_utils import events_to_midi_bytes, events_to_wav_bytes, piano_roll_figure

MODEL_PATH = "model.pt"

st.set_page_config(page_title="AI Music Generator", page_icon="🎵")
st.title("🎵 AI Music Generator")
st.caption("An LSTM neural network trained on MIDI music composes new note sequences.")

if not os.path.exists(MODEL_PATH):
    st.error("No trained model found. Run `python train.py` first, then restart this app.")
    st.stop()


@st.cache_resource
def get_model():
    return load_checkpoint(MODEL_PATH)


model, vocab, ids = get_model()

with st.sidebar:
    st.header("Settings")
    length = st.slider("Length (notes / chords)", 50, 400, 150, step=10)
    temperature = st.slider(
        "Creativity (temperature)", 0.3, 1.5, 0.9, step=0.05,
        help="Low = safe and repetitive. High = surprising but messier.",
    )
    top_k = st.slider("Top-k choices", 5, 50, 20)
    bpm = st.slider("Tempo (BPM)", 60, 180, 100)

if st.button("Generate music", type="primary"):
    with st.spinner("Composing..."):
        events = generate_events(model, vocab, ids, length, temperature, top_k)
        st.session_state["music"] = {
            "events": events,
            "midi": events_to_midi_bytes(events, bpm),
            "wav": events_to_wav_bytes(events, bpm),
            "seconds": sum(d for _, d in events) * 60 / bpm,
        }

if "music" in st.session_state:
    music = st.session_state["music"]
    st.subheader("Your composition")
    st.caption(f"{len(music['events'])} notes/chords, about {music['seconds']:.0f} seconds")
    st.audio(music["wav"], format="audio/wav")
    st.pyplot(piano_roll_figure(music["events"]))

    col1, col2 = st.columns(2)
    col1.download_button("Download MIDI", music["midi"], "generated.mid", "audio/midi")
    col2.download_button("Download WAV", music["wav"], "generated.wav", "audio/wav")
else:
    st.info("Pick your settings in the sidebar and click **Generate music**.")
