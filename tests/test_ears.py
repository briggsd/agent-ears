import json

import numpy as np
import pytest
import soundfile as sf

from agent_ears.audio import load
from agent_ears.cli import main
from agent_ears.metrics import analyze
from agent_ears.store import fit

SR = 24000


def voice(seconds: float, f0: float) -> np.ndarray:
    """Harmonic tone with a 4 Hz syllable-like amplitude envelope."""
    t = np.arange(int(seconds * SR)) / SR
    tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    return 0.2 * tone / np.max(np.abs(tone)) * (0.55 + 0.45 * np.sin(2 * np.pi * 4 * t))


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR))


def write(path, *parts) -> str:
    sf.write(path, np.concatenate(parts), SR, subtype="PCM_16")
    return str(path)


def test_pitch_pauses_and_silence(tmp_path):
    path = write(tmp_path / "a.wav", silence(0.2), voice(1.5, 120), silence(0.5), voice(1.5, 120), silence(0.4))
    result = analyze(load(path))
    m = result["metrics"]
    assert m["duration_s"] == pytest.approx(4.1, abs=0.01)
    assert m["f0_median_hz"] == pytest.approx(120, rel=0.02)
    assert m["leading_silence_s"] == pytest.approx(0.2, abs=0.04)
    assert m["trailing_silence_s"] == pytest.approx(0.4, abs=0.04)
    assert m["ends_abruptly"] == 0
    assert m["pause_count"] == 1
    assert result["pauses"][0]["start_s"] == pytest.approx(1.7, abs=0.05)
    assert result["pauses"][0]["duration_s"] == pytest.approx(0.5, abs=0.05)
    assert m["speech_rate_sps"] == pytest.approx(12 / 3.5, rel=0.25)


def test_abrupt_end_and_pitch_shift(tmp_path):
    low = analyze(load(write(tmp_path / "low.wav", voice(2, 100))))["metrics"]
    high = analyze(load(write(tmp_path / "high.wav", voice(2, 200))))["metrics"]
    assert low["ends_abruptly"] == 1
    assert high["f0_median_hz"] / low["f0_median_hz"] == pytest.approx(2, rel=0.03)
    assert high["centroid_hz"] > low["centroid_hz"]


def test_silent_file_is_an_error(tmp_path, capsys):
    path = write(tmp_path / "s.wav", silence(1))
    assert main(["report", path, "--no-models"]) == 2
    assert "silent" in capsys.readouterr().err


def test_verdicts_fit_and_diff(tmp_path, capsys):
    store = str(tmp_path / "store")
    high = [write(tmp_path / f"h{i}.wav", voice(2, 180 + 5 * i), silence(0.2)) for i in range(3)]
    low = [write(tmp_path / f"l{i}.wav", voice(2, 100 + 5 * i), silence(0.2)) for i in range(3)]
    for h, l in zip(high, low):
        assert main(["verdict", h, l, "--note", "brighter", "--no-models", "--store", store]) == 0
    assert main(["verdict", high[0], high[1], "--tie", "--no-models", "--store", store]) == 0
    capsys.readouterr()

    assert main(["fit", "--json", "--store", store]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["verdicts"], out["decided"]) == (4, 3)
    f0 = next(r for r in out["fit"] if r["key"] == "f0_median_hz")
    assert (f0["prefers"], f0["agree"], f0["n"]) == ("higher", 3, 3)

    assert main(["diff", low[0], high[0], "--json", "--no-models", "--store", store]) == 0
    diff = {d["key"]: d for d in json.loads(capsys.readouterr().out)["diff"]}
    assert diff["f0_median_hz"]["notable"] is True
    assert diff["f0_median_hz"]["picks"] == "higher 3/3"
    assert diff["duration_s"]["notable"] is False


def test_fit_ignores_ties_and_missing_values():
    def v(w, l, tie=False):
        return {"tie": tie, "winner": {"metrics": {"hnr_db": w}}, "loser": {"metrics": {"hnr_db": l}}}

    rows = fit([v(10, 8), v(9, 12), v(11, 7), v(1, 99, tie=True), v(None, 5)])
    assert [(r["key"], r["prefers"], r["agree"], r["n"]) for r in rows] == [("hnr_db", "higher", 2, 3)]


def test_plot_writes_png(tmp_path):
    path = write(tmp_path / "a.wav", voice(1.5, 120), silence(0.3))
    out = tmp_path / "a.png"
    assert main(["plot", path, path, "--tail", "1", "-o", str(out)]) == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
