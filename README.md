# CodeAlpha_MusicGeneration

Music generation with an LSTM neural network, built for the CodeAlpha AI Internship (Task 3).
The model learns note patterns from MIDI files and composes new music, which you can play in the browser or download as MIDI / WAV.

## How it works
1. **Collect MIDI data:** classical, jazz or any other `.mid` files go in the `midi_data/` folder.
2. **Preprocess** (`preprocess.py`): each piece is read with `mido` and turned into a sequence of *events* (the notes or chord played together, plus how many beats until the next event). Every piece is transposed to C major / A minor with key detection, so the model learns one common key.
3. **Build the model** (`model.py`): an embedding layer, a 2-layer LSTM and a linear layer that predicts the next event.
4. **Train** (`train.py`): the model learns to predict the next event from the previous 48, with a validation split to pick the best epoch.
5. **Generate** (`generate.py`): start from a short seed taken from the training music, then repeatedly sample the next event (with temperature and top-k control).
6. **Convert to audio:** events are written to a `.mid` file and rendered to a `.wav` file with a built-in synthesizer, so no extra software is needed.

## Setup
```bash
pip install -r requirements.txt
```
If `pip install torch` fails on a very new Python version, install Python 3.12 and use that instead.

## Run
```bash
# 1. Put MIDI files in midi_data/  (or create demo files: python make_sample_data.py)
# 2. Check your data
python preprocess.py midi_data
# 3. Train (a few minutes on CPU)
python train.py --epochs 30
# 4a. Generate from the command line
python generate.py --length 200 --temperature 0.9
# 4b. ...or use the web app
python -m streamlit run app.py
```

## Where to get MIDI data
Download 30-100 files of one style (for example a classical composer) and place them in `midi_data/`. Free sources include the "Classical Music MIDI" dataset on Kaggle and free classical MIDI sites. Check each source's license. `make_sample_data.py` only creates rule-based demo files for testing the pipeline.

## Tech Stack
Python, PyTorch, mido, NumPy, Matplotlib, Streamlit

## Settings in the app
- **Creativity (temperature):** low is safe and repetitive, high is surprising but messier.
- **Top-k:** how many of the most likely next events can be chosen.
- **Tempo, length:** control the speed and size of the piece.

## Screenshots
_Add screenshots here._

## Author
Varshaparamala | CodeAlpha AI Intern
