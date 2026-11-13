"""Human listening verdicts and the fit that turns them into metric directions.

The store is a directory (default ./.ears): verdicts.jsonl plus config.json.
Each verdict snapshots both files' metrics, so the fit works after the audio is gone.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import binomtest

from .metrics import SPECS


def store_dir(explicit: str | None = None) -> Path:
    return Path(explicit or os.environ.get("EARS_STORE") or ".ears").expanduser().resolve()


def read_config(store: Path) -> dict:
    path = store / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def write_config(store: Path, config: dict) -> None:
    store.mkdir(parents=True, exist_ok=True)
    (store / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def add_verdict(store: Path, winner: dict, loser: dict, note: str = "", tie: bool = False) -> dict:
    """`winner` and `loser` are report dicts (path, sha256, metrics). For a tie the order carries no meaning."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tie": tie,
        "note": note,
        "winner": {k: winner[k] for k in ("path", "sha256", "metrics")},
        "loser": {k: loser[k] for k in ("path", "sha256", "metrics")},
    }
    store.mkdir(parents=True, exist_ok=True)
    with open(store / "verdicts.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def read_verdicts(store: Path) -> list[dict]:
    path = store / "verdicts.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fit(verdicts: list[dict]) -> list[dict]:
    """Per metric: in how many decided pairs the winner had the higher value.

    Returns rows sorted by strength of agreement. `p` is a two-sided sign test;
    it cannot fall below 0.05 with fewer than 6 informative pairs.
    """
    rows = []
    for spec in SPECS:
        higher = lower = 0
        deltas = []
        for v in verdicts:
            if v.get("tie"):
                continue
            w, l = v["winner"]["metrics"].get(spec.key), v["loser"]["metrics"].get(spec.key)
            if w is None or l is None or w == l:
                continue
            deltas.append(w - l)
            if w > l:
                higher += 1
            else:
                lower += 1
        n = higher + lower
        if n == 0:
            continue
        deltas.sort()
        rows.append({
            "key": spec.key,
            "prefers": "higher" if higher >= lower else "lower",
            "agree": max(higher, lower),
            "n": n,
            "p": round(binomtest(max(higher, lower), n, 0.5).pvalue, 4),
            "median_delta": round(deltas[len(deltas) // 2], 4),
            "unit": spec.unit,
        })
    rows.sort(key=lambda r: (r["p"], -r["n"]))
    return rows
