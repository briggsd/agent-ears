"""Audio loading. Every file is analysed as mono float at ANALYSIS_SR so numbers compare across files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ANALYSIS_SR = 24000


@dataclass
class Audio:
    path: str
    sha256: str
    native_sr: int
    channels: int
    samples: np.ndarray  # mono float64 at ANALYSIS_SR
    sr: int = ANALYSIS_SR

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sr


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    frac = Fraction(sr_to, sr_from)
    return resample_poly(x, frac.numerator, frac.denominator)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: str | Path) -> Audio:
    p = Path(path).expanduser().resolve()
    data, sr = sf.read(p, dtype="float64", always_2d=True)
    if data.shape[0] == 0:
        raise ValueError(f"{p}: no audio samples")
    if not np.isfinite(data).all():
        raise ValueError(f"{p}: non-finite samples")
    mono = data.mean(axis=1)
    return Audio(
        path=str(p),
        sha256=sha256_file(p),
        native_sr=int(sr),
        channels=data.shape[1],
        samples=resample(mono, int(sr), ANALYSIS_SR),
    )
