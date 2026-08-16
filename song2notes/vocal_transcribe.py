"""Vocal melody transcription: isolated vocal audio -> MIDI notes.

Uses Spotify's basic-pitch pretrained pitch-detection model.
"""

from __future__ import annotations

from pathlib import Path

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict_and_save


def vocals_to_midi(vocal_wav: Path, out_dir: Path, out_name: str = "vocals_notes") -> Path:
    """Transcribe a vocal-only audio file to a MIDI file and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)

    predict_and_save(
        [str(vocal_wav)],
        str(out_dir),
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
    )

    generated = out_dir / f"{vocal_wav.stem}_basic_pitch.mid"
    final_path = out_dir / f"{out_name}.mid"
    generated.replace(final_path)
    return final_path
