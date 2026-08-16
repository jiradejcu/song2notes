"""Drum hit transcription: isolated drum audio -> MIDI notes (General MIDI drum map).

There is no widely available, reliably pip-installable pretrained model for
multi-class drum transcription, so this module uses classical onset detection
(librosa) followed by a frequency-band heuristic classifier: each onset is
bucketed into kick / snare / closed hi-hat / open hi-hat by where its energy
sits in the spectrum and how quickly it decays. This is intentionally simple
and will make mistakes on busy kits or heavily processed drum mixes.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pretty_midi

# General MIDI percussion note numbers (channel 10).
GM_KICK = 36
GM_SNARE = 38
GM_HIHAT_CLOSED = 42
GM_HIHAT_OPEN = 46

_LOW_BAND = (20, 120)
_MID_BAND = (120, 2000)
_HIGH_BAND = (2000, 8000)

_ANALYSIS_WINDOW_S = 0.08
_SUSTAIN_WINDOW_S = 0.25


def _band_energy(magnitudes: np.ndarray, freqs: np.ndarray, band: tuple[float, float]) -> float:
    mask = (freqs >= band[0]) & (freqs < band[1])
    if not np.any(mask):
        return 0.0
    return float(np.sum(magnitudes[mask] ** 2))


def _classify_onset(y: np.ndarray, sr: int, onset_sample: int) -> tuple[int, int]:
    """Return (GM drum note, MIDI velocity) for the hit starting at onset_sample."""
    win = int(_ANALYSIS_WINDOW_S * sr)
    frame = y[onset_sample : onset_sample + win]
    if len(frame) < win:
        frame = np.pad(frame, (0, win - len(frame)))

    spectrum = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(len(frame), d=1.0 / sr)

    low = _band_energy(spectrum, freqs, _LOW_BAND)
    mid = _band_energy(spectrum, freqs, _MID_BAND)
    high = _band_energy(spectrum, freqs, _HIGH_BAND)
    total = low + mid + high + 1e-9

    peak_amp = float(np.max(np.abs(frame)))
    velocity = int(np.clip(peak_amp * 400, 40, 127))

    if low / total > 0.5:
        note = GM_KICK
    elif high / total > 0.4:
        sustain_win = int(_SUSTAIN_WINDOW_S * sr)
        tail = y[onset_sample : onset_sample + sustain_win]
        early = np.abs(tail[: win]).mean() if len(tail) >= win else peak_amp
        late = np.abs(tail[-win:]).mean() if len(tail) >= win else 0.0
        decay_ratio = late / (early + 1e-9)
        note = GM_HIHAT_OPEN if decay_ratio > 0.35 else GM_HIHAT_CLOSED
    else:
        note = GM_SNARE

    return note, velocity


def drums_to_midi(drum_wav: Path, out_dir: Path, out_name: str = "drums_notes") -> Path:
    """Transcribe a drums-only audio file to a MIDI drum track and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(str(drum_wav), sr=None, mono=True)
    onset_samples = librosa.onset.onset_detect(
        y=y, sr=sr, units="samples", backtrack=True, hop_length=256
    )

    midi = pretty_midi.PrettyMIDI()
    drum_track = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")

    hit_duration_s = 0.1
    for onset_sample in onset_samples:
        note_number, velocity = _classify_onset(y, sr, int(onset_sample))
        start = onset_sample / sr
        drum_track.notes.append(
            pretty_midi.Note(
                velocity=velocity,
                pitch=note_number,
                start=start,
                end=start + hit_duration_s,
            )
        )

    midi.instruments.append(drum_track)
    out_path = out_dir / f"{out_name}.mid"
    midi.write(str(out_path))
    return out_path
