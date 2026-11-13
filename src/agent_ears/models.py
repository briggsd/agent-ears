"""Model-backed metrics: DNSMOS quality prediction and speaker-embedding similarity.

Weights download once to ~/.cache/agent-ears/models (override with EARS_CACHE) and are hash-checked.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

from .audio import Audio, resample

MODEL_SR = 16000

MODELS = {
    "dnsmos": (
        "sig_bak_ovr.onnx",
        "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
        "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd",
    ),
    "speaker": (
        "wespeaker_en_voxceleb_resnet34.onnx",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx",
        "5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94",
    ),
}


def cache_dir() -> Path:
    base = os.environ.get("EARS_CACHE") or Path.home() / ".cache" / "agent-ears"
    return Path(base) / "models"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_path(name: str) -> Path:
    filename, url, sha = MODELS[name]
    path = cache_dir() / filename
    if path.exists() and _sha256(path) == sha:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"ears: downloading {filename}", file=sys.stderr)
    tmp = path.with_suffix(f".part{os.getpid()}")
    try:
        urllib.request.urlretrieve(url, tmp)
        got = _sha256(tmp)
        if got != sha:
            raise RuntimeError(f"{filename}: sha256 {got} does not match pinned {sha}")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _at_16k(audio: Audio) -> np.ndarray:
    return resample(audio.samples, audio.sr, MODEL_SR).astype(np.float32)


@lru_cache(maxsize=1)
def _dnsmos_session():
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(model_path("dnsmos")), options, providers=["CPUExecutionProvider"])


_DNSMOS_LEN = int(9.01 * MODEL_SR)
# Polynomial calibration from Microsoft's dnsmos_local.py (non-personalized model).
_P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
_P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
_P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])


def dnsmos(audio: Audio) -> dict[str, float]:
    """Predicted MOS (1-5). Trained for noise-suppression quality, so treat it as a rough proxy for TTS."""
    session = _dnsmos_session()
    x = _at_16k(audio)
    while len(x) < _DNSMOS_LEN:
        x = np.append(x, x)
    name = session.get_inputs()[0].name
    # One-second hops as in Microsoft's script, plus a last window flush with the end so the tail is scored.
    starts = list(range(0, len(x) - _DNSMOS_LEN + 1, MODEL_SR))
    if starts[-1] != len(x) - _DNSMOS_LEN:
        starts.append(len(x) - _DNSMOS_LEN)
    rows = [session.run(None, {name: x[s : s + _DNSMOS_LEN][None, :]})[0][0] for s in starts]
    sig, bak, ovr = np.mean(rows, axis=0)
    return {
        "dnsmos_sig": round(float(_P_SIG(sig)), 4),
        "dnsmos_bak": round(float(_P_BAK(bak)), 4),
        "dnsmos_ovr": round(float(_P_OVR(ovr)), 4),
    }


@lru_cache(maxsize=1)
def _speaker_extractor():
    import sherpa_onnx

    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model_path("speaker")), num_threads=2)
    return sherpa_onnx.SpeakerEmbeddingExtractor(config)


def speaker_embedding(audio: Audio) -> np.ndarray:
    extractor = _speaker_extractor()
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=MODEL_SR, waveform=_at_16k(audio))
    stream.input_finished()
    emb = np.asarray(extractor.compute(stream), dtype=np.float64)
    return emb / np.linalg.norm(emb)


def speaker_similarity(audio: Audio, reference: Audio) -> float:
    return round(float(np.dot(speaker_embedding(audio), speaker_embedding(reference))), 4)
