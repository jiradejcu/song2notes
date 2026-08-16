from __future__ import annotations

import argparse
import sys
from pathlib import Path

from song2notes.pipeline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="song2notes",
        description="Extract vocal melody and drum hit notes (as MIDI) from a song file.",
    )
    parser.add_argument("input", type=Path, help="Path to the input song audio file (wav/mp3/etc).")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("song2notes_output"),
        help="Directory to write output MIDI files to (default: ./song2notes_output).",
    )
    parser.add_argument(
        "--keep-stems",
        action="store_true",
        help="Keep the intermediate separated vocals.wav/drums.wav stem files.",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(f"input file not found: {args.input}")

    print(f"Separating stems and transcribing notes from {args.input} ...")
    result = run(args.input, args.output_dir, keep_stems=args.keep_stems)

    print(f"Vocal melody notes -> {result.vocals_midi}")
    print(f"Drum hit notes     -> {result.drums_midi}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
