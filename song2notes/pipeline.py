from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from song2notes.drum_transcribe import drums_to_midi
from song2notes.separate import separate
from song2notes.vocal_transcribe import vocals_to_midi


@dataclass
class Result:
    vocals_midi: Path
    drums_midi: Path
    vocals_stem: Path
    drums_stem: Path


def run(input_path: Path, output_dir: Path, keep_stems: bool = False) -> Result:
    """Separate ``input_path`` into stems and transcribe vocal + drum notes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stems_dir = output_dir / "stems"

    stems = separate(input_path, stems_dir)
    vocals_midi = vocals_to_midi(stems.vocals, output_dir)
    drums_midi = drums_to_midi(stems.drums, output_dir)

    if not keep_stems:
        for stem_path in (stems.vocals, stems.drums, stems.bass, stems.other):
            stem_path.unlink(missing_ok=True)

    return Result(
        vocals_midi=vocals_midi,
        drums_midi=drums_midi,
        vocals_stem=stems.vocals,
        drums_stem=stems.drums,
    )
