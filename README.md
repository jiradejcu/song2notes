# song2notes

Command-line tool that takes a song audio file and outputs two MIDI files:
the vocal melody as note events, and the drum hits mapped onto the General
MIDI drum map.

## How it works

1. **Separation** — [Demucs](https://github.com/facebookresearch/demucs)
   (`htdemucs` model) splits the input mix into `vocals`, `drums`, `bass`,
   and `other` stems.
2. **Vocal transcription** — the isolated vocal stem is run through
   Spotify's [basic-pitch](https://github.com/spotify/basic-pitch) pretrained
   pitch-detection model, producing a MIDI file of the sung melody.
3. **Drum transcription** — the isolated drum stem is analyzed with
   [librosa](https://librosa.org/) onset detection, then each onset is
   classified into kick / snare / closed hi-hat / open hi-hat by a
   frequency-band + decay heuristic (there is no robust, easily
   pip-installable pretrained model for multi-class drum transcription).
   This is approximate: busy patterns, heavily processed drums, or hits
   that land on the same beat as another drum can be misclassified.

## Setup

```bash
cd song2notes
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Demucs and basic-pitch ship/download their own pretrained model weights on
first run — the first invocation will be slower and needs internet access.

## Usage

```bash
python -m song2notes path/to/song.mp3 -o output_dir/
```

Output:

```
output_dir/
├── vocals_notes.mid   # vocal melody as MIDI notes
└── drums_notes.mid    # drum hits as a MIDI drum track (channel 10)
```

Pass `--keep-stems` to also keep the intermediate separated
`vocals.wav` / `drums.wav` / `bass.wav` / `other.wav` files under
`output_dir/stems/`.

## Generating a test file

No copyrighted audio is bundled. `tests/generate_sample_song.py` synthesizes
a short sine-wave melody over a simple drum pattern for smoke-testing the
pipeline:

```bash
python tests/generate_sample_song.py
python -m song2notes tests/fixtures/sample_song.wav -o /tmp/out
```

## Limitations

- CPU-only inference; a multi-minute song can take a minute or more to
  separate and transcribe.
- basic-pitch expects a single melodic line; it will produce odd results on
  vocal stems with heavy harmonies or background vocals left in the mix.
- The drum classifier is a hand-tuned heuristic, not a trained model — treat
  its note pitches (kick/snare/hi-hat) as a best-effort guess, and the onset
  timing as the more reliable signal.
