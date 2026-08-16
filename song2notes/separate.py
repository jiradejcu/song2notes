"""Source separation: split a song into vocals/drums/bass/other stems using Demucs."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEMUCS_MODEL = "htdemucs"


@dataclass
class Stems:
    vocals: Path
    drums: Path
    bass: Path
    other: Path


def separate(input_path: Path, work_dir: Path) -> Stems:
    """Run Demucs on ``input_path`` and return paths to the separated stems.

    Stems are written under ``work_dir/<model>/<track_name>/*.wav``.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "demucs",
            "-n",
            DEMUCS_MODEL,
            "--out",
            str(work_dir),
            str(input_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Demucs separation failed (exit {result.returncode}):\n{result.stderr}"
        )

    track_name = input_path.stem
    stem_dir = work_dir / DEMUCS_MODEL / track_name
    stems = Stems(
        vocals=stem_dir / "vocals.wav",
        drums=stem_dir / "drums.wav",
        bass=stem_dir / "bass.wav",
        other=stem_dir / "other.wav",
    )
    for name, path in (
        ("vocals", stems.vocals),
        ("drums", stems.drums),
        ("bass", stems.bass),
        ("other", stems.other),
    ):
        if not path.exists():
            raise RuntimeError(f"Expected Demucs {name} stem missing: {path}")
    return stems
