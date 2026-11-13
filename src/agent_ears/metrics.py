"""Acoustic metrics for one speech file. All local, no model weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import parselmouth
import pyloudnorm
from parselmouth.praat import call
from scipy.signal import find_peaks, stft

from .audio import Audio

HOP_S = 0.01
WIN_S = 0.025
MIN_PAUSE_S = 0.15


@dataclass(frozen=True)
class Spec:
    key: str
    unit: str
    notable: float  # a delta at least this large is flagged in diffs
    relative: bool  # notable is a fraction of the first value instead of an absolute delta
    desc: str


# The `notable` thresholds are rough audibility guesses, not measured limits.
SPECS: list[Spec] = [
    Spec("duration_s", "s", 0.03, True, "total file length"),
    Spec("lufs", "LUFS", 1.0, False, "integrated loudness"),
    Spec("peak_dbfs", "dBFS", 1.0, False, "sample peak"),
    Spec("clipped_samples", "n", 1, False, "samples at full scale"),
    Spec("level_sd_db", "dB", 1.0, False, "loudness variation across speech frames; low reads as flat delivery"),
    Spec("leading_silence_s", "s", 0.05, False, "silence before first speech"),
    Spec("trailing_silence_s", "s", 0.05, False, "silence after last speech"),
    Spec("ends_abruptly", "0/1", 1, False, "1 when speech energy runs to the last 30 ms of the file"),
    Spec("pause_count", "n", 1, False, f"internal pauses of {MIN_PAUSE_S}s or longer"),
    Spec("pause_total_s", "s", 0.15, False, "summed internal pause time"),
    Spec("longest_pause_s", "s", 0.10, False, "longest internal pause"),
    Spec("speech_rate_sps", "syll/s", 0.3, False, "estimated syllables per second including pauses"),
    Spec("articulation_rate_sps", "syll/s", 0.3, False, "estimated syllables per second excluding pauses"),
    Spec("f0_median_hz", "Hz", 0.06, True, "median pitch; 6% is about one semitone"),
    Spec("f0_range_st", "st", 1.5, False, "pitch range, 5th to 95th percentile, in semitones"),
    Spec("f0_sd_st", "st", 0.5, False, "pitch variation in semitones; low reads as monotone"),
    Spec("f0_final_slope_st_s", "st/s", 5.0, False, "pitch slope over the last 0.5 s of speech; negative is a falling ending"),
    Spec("voiced_fraction", "", 0.05, False, "share of speech frames with pitch"),
    Spec("jitter_pct", "%", 0.3, False, "cycle-to-cycle pitch instability; higher reads as rough"),
    Spec("shimmer_pct", "%", 1.0, False, "cycle-to-cycle amplitude instability; higher reads as rough"),
    Spec("hnr_db", "dB", 1.0, False, "harmonics-to-noise ratio; higher reads as clearer, lower as breathy or hoarse"),
    Spec("cpps_db", "dB", 1.0, False, "smoothed cepstral peak prominence; higher reads as a stronger, cleaner voice"),
    Spec("centroid_hz", "Hz", 0.08, True, "spectral centre of mass over speech; higher reads as brighter"),
    Spec("alpha_ratio_db", "dB", 1.5, False, "energy in 1-5 kHz relative to 50 Hz-1 kHz; higher reads as brighter or more pressed"),
    Spec("hf_ratio_db", "dB", 2.0, False, "energy above 5 kHz relative to below; higher reads as more air or sibilance"),
    Spec("dnsmos_ovr", "MOS", 0.15, False, "DNSMOS predicted overall quality, 1-5"),
    Spec("dnsmos_sig", "MOS", 0.15, False, "DNSMOS predicted speech signal quality, 1-5"),
    Spec("dnsmos_bak", "MOS", 0.15, False, "DNSMOS predicted background cleanliness, 1-5"),
    Spec("speaker_sim", "cos", 0.05, False, "speaker-embedding cosine similarity to the reference voice"),
]
SPEC_BY_KEY = {s.key: s for s in SPECS}


def frame_db(x: np.ndarray, sr: int) -> np.ndarray:
    """RMS level in dBFS per 10 ms hop."""
    hop = int(round(HOP_S * sr))
    win = int(round(WIN_S * sr))
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-10))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def speech_mask(db: np.ndarray) -> tuple[np.ndarray, float]:
    threshold = max(float(np.percentile(db, 95)) - 30.0, -60.0)
    return db > threshold, threshold


def analyze(audio: Audio, f0_min: float = 60.0, f0_max: float = 400.0) -> dict:
    """Return {'metrics': {key: number|None}, 'pauses': [{'start_s','duration_s'}]}."""
    x, sr = audio.samples, audio.sr
    m: dict[str, float | int | None] = {s.key: None for s in SPECS}
    m["duration_s"] = audio.duration

    peak = float(np.max(np.abs(x)))
    if peak < 1e-4:
        raise ValueError(f"{audio.path}: file is silent")
    m["peak_dbfs"] = 20 * np.log10(peak)
    m["clipped_samples"] = int(np.sum(np.abs(x) >= 0.999))
    if audio.duration >= 0.5:
        m["lufs"] = float(pyloudnorm.Meter(sr).integrated_loudness(x))

    db = frame_db(x, sr)
    speech, _ = speech_mask(db)
    idx = np.flatnonzero(speech)
    first, last = int(idx[0]), int(idx[-1])
    m["leading_silence_s"] = first * HOP_S
    speech_end_s = min(last * HOP_S + WIN_S, audio.duration)
    m["trailing_silence_s"] = audio.duration - speech_end_s
    m["ends_abruptly"] = int(m["trailing_silence_s"] < 0.03)
    m["level_sd_db"] = float(np.std(db[speech]))

    pauses = [
        {"start_s": round(a * HOP_S, 2), "duration_s": round((b - a) * HOP_S, 2)}
        for a, b in _runs(~speech[first : last + 1])
        if (b - a) * HOP_S >= MIN_PAUSE_S
    ]
    for p in pauses:
        p["start_s"] = round(p["start_s"] + first * HOP_S, 2)
    pause_total = sum(p["duration_s"] for p in pauses)
    m["pause_count"] = len(pauses)
    m["pause_total_s"] = pause_total
    m["longest_pause_s"] = max((p["duration_s"] for p in pauses), default=0.0)

    snd = parselmouth.Sound(x, sampling_frequency=sr)
    pitch = snd.to_pitch_ac(time_step=HOP_S, pitch_floor=f0_min, pitch_ceiling=f0_max)
    f0 = pitch.selected_array["frequency"]
    f0_t = pitch.xs()
    voiced = f0 > 0

    span_s = speech_end_s - first * HOP_S
    intensity = snd.to_intensity(minimum_pitch=50.0, time_step=HOP_S)
    iv, it = intensity.values[0], intensity.xs()
    peaks, _ = find_peaks(iv, height=np.percentile(iv, 99) - 25.0, prominence=2.0)
    syllables = sum(1 for p in peaks if not np.isnan(pitch.get_value_at_time(it[p])))
    if span_s > 0:
        m["speech_rate_sps"] = syllables / span_s
    if span_s - pause_total > 0:
        m["articulation_rate_sps"] = syllables / (span_s - pause_total)

    m["voiced_fraction"] = float(np.sum(voiced) * HOP_S / max(np.sum(speech) * HOP_S, HOP_S))
    if np.sum(voiced) >= 10:
        st = 12 * np.log2(f0[voiced] / 100.0)
        m["f0_median_hz"] = float(np.median(f0[voiced]))
        m["f0_range_st"] = float(np.percentile(st, 95) - np.percentile(st, 5))
        m["f0_sd_st"] = float(np.std(st))
        tail = voiced & (f0_t >= speech_end_s - 0.5)
        if np.sum(tail) >= 5:
            m["f0_final_slope_st_s"] = float(np.polyfit(f0_t[tail], 12 * np.log2(f0[tail] / 100.0), 1)[0])

        points = call(snd, "To PointProcess (periodic, cc)", f0_min, f0_max)
        m["jitter_pct"] = 100 * call(points, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        m["shimmer_pct"] = 100 * call([snd, points], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harmonicity = call(snd, "To Harmonicity (cc)", HOP_S, f0_min, 0.1, 1.0)
        m["hnr_db"] = call(harmonicity, "Get mean", 0, 0)
        cepstrogram = call(snd, "To PowerCepstrogram", f0_min, 0.002, 5000, 50)
        m["cpps_db"] = call(
            cepstrogram, "Get CPPS", False, 0.01, 0.001, f0_min, f0_max, 0.05,
            "Parabolic", 0.001, 0, "Straight", "Robust",
        )

    freqs, times, spec = stft(x, fs=sr, nperseg=1024, noverlap=768)
    frame_of = np.clip(np.round(times / HOP_S).astype(int), 0, len(speech) - 1)
    power = (np.abs(spec) ** 2)[:, speech[frame_of]].mean(axis=1)

    def band(lo: float, hi: float) -> float:
        return float(power[(freqs >= lo) & (freqs < hi)].sum())

    low, mid, high = band(50, 1000), band(1000, 5000), band(5000, sr / 2)
    usable = freqs >= 50
    m["centroid_hz"] = float((freqs[usable] * power[usable]).sum() / power[usable].sum())
    m["alpha_ratio_db"] = 10 * np.log10(max(mid, 1e-20) / max(low, 1e-20))
    if audio.native_sr >= 16000:
        m["hf_ratio_db"] = 10 * np.log10(max(high, 1e-20) / max(low + mid, 1e-20))

    for k, v in m.items():
        if isinstance(v, (float, np.floating)):
            m[k] = None if not np.isfinite(v) else round(float(v), 4)
    return {"metrics": m, "pauses": pauses}
