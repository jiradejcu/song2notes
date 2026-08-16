"""Generate a small synthetic test song (sung-like melody + drum pattern) for
exercising the song2notes pipeline without needing a real copyrighted track.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
DURATION_S = 6.0
BPM = 120
BEAT_S = 60.0 / BPM


def _envelope(n: int, attack: int, release: int) -> np.ndarray:
    env = np.ones(n)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return env


def _kick(sr: int) -> np.ndarray:
    t = np.linspace(0, 0.25, int(0.25 * sr), endpoint=False)
    freq = 120 * np.exp(-t * 18)
    tone = np.sin(2 * np.pi * np.cumsum(freq) / sr)
    return tone * np.exp(-t * 14)


def _snare(sr: int) -> np.ndarray:
    t = np.linspace(0, 0.18, int(0.18 * sr), endpoint=False)
    noise = np.random.default_rng(1).standard_normal(len(t))
    tone = np.sin(2 * np.pi * 180 * t)
    return (0.6 * noise + 0.4 * tone) * np.exp(-t * 20)


def _hihat(sr: int, open_hat: bool = False) -> np.ndarray:
    length = 0.35 if open_hat else 0.06
    t = np.linspace(0, length, int(length * sr), endpoint=False)
    noise = np.random.default_rng(2).standard_normal(len(t))
    decay = 6 if open_hat else 40
    return noise * np.exp(-t * decay) * 0.5


def _mix_at(track: np.ndarray, hit: np.ndarray, start_s: float) -> None:
    start = int(start_s * SR)
    end = min(start + len(hit), len(track))
    track[start:end] += hit[: end - start]


def _vocal_melody(sr: int, duration_s: float) -> np.ndarray:
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False)
    note_freqs = [261.63, 329.63, 392.00, 329.63]  # C4 E4 G4 E4
    note_len = duration_s / len(note_freqs)
    freq = np.zeros(n)
    for i, f in enumerate(note_freqs):
        start = int(i * note_len * sr)
        end = int((i + 1) * note_len * sr)
        vibrato = 1.0 + 0.01 * np.sin(2 * np.pi * 5 * t[start:end])
        freq[start:end] = f * vibrato
    phase = 2 * np.pi * np.cumsum(freq) / sr
    tone = np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
    env = _envelope(n, attack=int(0.05 * sr), release=int(0.05 * sr))
    return tone * env * 0.5


def generate(out_path: Path) -> None:
    n = int(DURATION_S * SR)
    drums = np.zeros(n)

    beat = 0.0
    step = 0
    while beat < DURATION_S:
        if step % 4 == 0:
            _mix_at(drums, _kick(SR), beat)
        if step % 4 == 2:
            _mix_at(drums, _snare(SR), beat)
        _mix_at(drums, _hihat(SR, open_hat=(step % 8 == 6)), beat)
        beat += BEAT_S / 2  # eighth notes
        step += 1

    vocals = _vocal_melody(SR, DURATION_S)

    mix = 0.7 * vocals + 0.8 * drums
    mix = mix / max(1.0, np.max(np.abs(mix)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), mix.astype(np.float32), SR)


if __name__ == "__main__":
    generate(Path(__file__).parent / "fixtures" / "sample_song.wav")
    print("wrote tests/fixtures/sample_song.wav")
