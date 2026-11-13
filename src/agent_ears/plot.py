"""Render mel spectrogram, pitch and level for one or more files into a PNG an agent can read."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
from scipy.signal import stft

from .audio import Audio
from .metrics import HOP_S, frame_db, speech_mask

N_FFT = 1024
N_MELS = 96


def _mel_filterbank(sr: int) -> tuple[np.ndarray, np.ndarray]:
    def to_mel(f):
        return 2595 * np.log10(1 + np.asarray(f) / 700)

    def to_hz(m):
        return 700 * (10 ** (np.asarray(m) / 2595) - 1)

    edges = to_hz(np.linspace(to_mel(0), to_mel(sr / 2), N_MELS + 2))
    freqs = np.linspace(0, sr / 2, N_FFT // 2 + 1)
    bank = np.zeros((N_MELS, len(freqs)))
    for i in range(N_MELS):
        lo, mid, hi = edges[i : i + 3]
        bank[i] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None)
    return bank, edges[1:-1]


def plot(audios: list[Audio], out: Path, start: float = 0.0, end: float | None = None,
         f0_min: float = 60.0, f0_max: float = 400.0) -> Path:
    """Two rows per file on a shared time axis. `start`/`end` are seconds; a negative start counts from each file's end."""
    windows = []
    for a in audios:
        t0 = max(a.duration + start, 0.0) if start < 0 else start
        t1 = a.duration if end is None else min(end, a.duration)
        windows.append((t0, t1))
    span = max(t1 - t0 for t0, t1 in windows)
    width = float(np.clip(span * 0.6, 10, 32))
    fig, axes = plt.subplots(
        2 * len(audios), 1, figsize=(width, 4.6 * len(audios)), squeeze=False,
        gridspec_kw={"height_ratios": [3, 2] * len(audios)},
    )
    for i, (audio, (t0, t1)) in enumerate(zip(audios, windows)):
        ax_spec, ax_pitch = axes[2 * i, 0], axes[2 * i + 1, 0]
        x, sr = audio.samples, audio.sr
        hop = int(HOP_S * sr)
        _, times, spec = stft(x, fs=sr, nperseg=N_FFT, noverlap=N_FFT - hop)
        bank, centres = _mel_filterbank(sr)
        mel_db = 10 * np.log10(np.maximum(bank @ (np.abs(spec) ** 2), 1e-12))
        top = mel_db.max()
        ax_spec.imshow(
            mel_db, origin="lower", aspect="auto", cmap="magma", vmin=top - 80, vmax=top,
            extent=[times[0], times[-1], 0, N_MELS],
        )
        ticks = [100, 250, 500, 1000, 2000, 4000, 8000]
        ticks = [f for f in ticks if f < sr / 2]
        ax_spec.set_yticks([float(np.interp(f, centres, np.arange(N_MELS) + 0.5)) for f in ticks])
        ax_spec.set_yticklabels([f"{f}" for f in ticks])
        ax_spec.set_ylabel("Hz (mel)")
        ax_spec.set_title(f"{Path(audio.path).name}   {audio.duration:.2f}s", loc="left", fontsize=10)

        db = frame_db(x, sr)
        speech, threshold = speech_mask(db)
        db_t = np.arange(len(db)) * HOP_S
        pitch = parselmouth.Sound(x, sampling_frequency=sr).to_pitch_ac(
            time_step=HOP_S, pitch_floor=f0_min, pitch_ceiling=f0_max
        )
        f0 = pitch.selected_array["frequency"]
        f0[f0 == 0] = np.nan
        ax_pitch.plot(pitch.xs(), f0, ".", ms=2.5, color="tab:blue")
        ax_pitch.set_yscale("log")
        ax_pitch.set_ylim(f0_min, f0_max)
        ax_pitch.set_yticks([60, 80, 100, 150, 200, 300, 400])
        ax_pitch.set_yticklabels(["60", "80", "100", "150", "200", "300", "400"])
        ax_pitch.minorticks_off()
        ax_pitch.set_ylabel("f0 Hz", color="tab:blue")
        ax_pitch.grid(True, alpha=0.3)
        ax_level = ax_pitch.twinx()
        ax_level.plot(db_t, db, color="tab:gray", lw=0.8, alpha=0.8)
        ax_level.fill_between(db_t, -90, 0, where=~speech, color="tab:orange", alpha=0.18, step="mid")
        ax_level.set_ylim(-90, 0)
        ax_level.set_ylabel("level dBFS (orange = silence)", color="tab:gray")
        for ax in (ax_spec, ax_pitch):
            # Equal-width windows keep seconds visually comparable between files.
            ax.set_xlim(t0, t0 + span)
            ax.axvline(audio.duration, color="white" if ax is ax_spec else "black", lw=0.8, ls="--")
        ax_pitch.set_xlabel("seconds")
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out
