"""ears: acoustic reports, A/B diffs, spectrograms and human listening verdicts."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__
from .audio import Audio, load
from .metrics import SPEC_BY_KEY, SPECS, analyze
from .store import add_verdict, fit, read_config, read_verdicts, store_dir, write_config

MIN_VERDICTS_FOR_PICKS = 3


def build_report(path: str, ref: Audio | None, use_models: bool, f0_min: float, f0_max: float) -> dict:
    audio = load(path)
    result = analyze(audio, f0_min=f0_min, f0_max=f0_max)
    report = {
        "path": audio.path,
        "sha256": audio.sha256,
        "native_sr": audio.native_sr,
        "channels": audio.channels,
        "metrics": result["metrics"],
        "pauses": result["pauses"],
        "warnings": [],
    }
    if use_models:
        from . import models

        steps = [("dnsmos", lambda: report["metrics"].update(models.dnsmos(audio)))]
        if ref is not None:
            steps.append(("speaker_sim", lambda: report["metrics"].update(
                speaker_sim=models.speaker_similarity(audio, ref))))
        for name, step in steps:
            try:
                step()
            except Exception as e:  # a missing model must not hide the acoustic metrics
                report["warnings"].append(f"{name} unavailable: {e}")
                print(f"ears: warning: {name} unavailable: {e}", file=sys.stderr)
    return report


def resolve_ref(args, store: Path) -> Audio | None:
    path = args.ref or read_config(store).get("ref")
    return load(path) if path else None


def reports_for(args, paths: list[str]) -> list[dict]:
    store = store_dir(args.store)
    ref = None if args.no_models else resolve_ref(args, store)
    return [build_report(p, ref, not args.no_models, args.f0_min, args.f0_max) for p in paths]


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"


def labels_for(paths: list[str]) -> list[str]:
    names = [Path(p).name for p in paths]
    if len(set(names)) == len(names):
        return names
    return [str(Path(*Path(p).parts[-2:])) for p in paths]


def print_table(rows: list[list[str]]) -> None:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))).rstrip())


def cmd_report(args) -> int:
    reports = reports_for(args, args.files)
    if args.json:
        print(json.dumps(reports if len(reports) > 1 else reports[0], indent=2))
        return 0
    rows = [["metric", *labels_for([r["path"] for r in reports]), "unit"]]
    for spec in SPECS:
        values = [r["metrics"].get(spec.key) for r in reports]
        if any(v is not None for v in values):
            rows.append([spec.key, *map(fmt, values), spec.unit])
    print_table(rows)
    for r, label in zip(reports, rows[0][1:]):
        if r["pauses"]:
            print(f"pauses {label}: " + ", ".join(f"{p['start_s']:.2f}s+{p['duration_s']:.2f}" for p in r["pauses"]))
    return 0


def notable(spec, a, b) -> bool:
    if a is None or b is None:
        return False
    limit = spec.notable * abs(a) if spec.relative else spec.notable
    return abs(b - a) >= limit


def cmd_diff(args) -> int:
    a, b = reports_for(args, [args.a, args.b])
    verdicts = [v for v in read_verdicts(store_dir(args.store)) if not v.get("tie")]
    # Only show directions the human's picks agree on in at least 80% of 3 or more pairs.
    picks = {r["key"]: r for r in fit(verdicts) if r["n"] >= MIN_VERDICTS_FOR_PICKS and r["agree"] / r["n"] >= 0.8}
    out = []
    for spec in SPECS:
        va, vb = a["metrics"].get(spec.key), b["metrics"].get(spec.key)
        if va is None and vb is None:
            continue
        delta = None if va is None or vb is None else round(vb - va, 4)
        pick = picks.get(spec.key)
        out.append({
            "key": spec.key, "a": va, "b": vb, "delta": delta, "unit": spec.unit,
            "notable": notable(spec, va, vb),
            "picks": f"{pick['prefers']} {pick['agree']}/{pick['n']}" if pick else None,
        })
    if args.json:
        print(json.dumps({"a": a, "b": b, "diff": out}, indent=2))
        return 0
    la, lb = labels_for([a["path"], b["path"]])
    header = ["metric", f"A {la}", f"B {lb}", "B-A", "", "unit"] + (["your picks"] if picks else [])
    rows = [header]
    for d in out:
        delta = "-" if d["delta"] is None else f"{d['delta']:+.3f}" if abs(d["delta"]) < 10 else f"{d['delta']:+.1f}"
        row = [d["key"], fmt(d["a"]), fmt(d["b"]), delta, "*" if d["notable"] else "", d["unit"]]
        rows.append(row + ([d["picks"] or ""] if picks else []))
    print_table(rows)
    print("* = delta at or above the metric's notable threshold (see `ears metrics`)")
    return 0


def cmd_plot(args) -> int:
    from .plot import plot

    start = -args.tail if args.tail else args.start
    out = plot([load(p) for p in args.files], Path(args.out), start=start, end=args.end,
               f0_min=args.f0_min, f0_max=args.f0_max)
    print(out.resolve())
    return 0


def cmd_metrics(args) -> int:
    if args.json:
        print(json.dumps([s.__dict__ for s in SPECS], indent=2))
        return 0
    rows = [["metric", "unit", "notable", "meaning"]]
    for s in SPECS:
        rows.append([s.key, s.unit, f"{s.notable:g}{' rel' if s.relative else ''}", s.desc])
    widths = [max(len(r[i]) for r in rows) for i in range(3)]
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r[:3], widths)) + "  " + r[3])
    return 0


def cmd_verdict(args) -> int:
    winner, loser = reports_for(args, [args.winner, args.loser])
    store = store_dir(args.store)
    add_verdict(store, winner, loser, note=args.note, tie=args.tie)
    total = len(read_verdicts(store))
    relation = "tied with" if args.tie else "beats"
    print(f"recorded: {Path(winner['path']).name} {relation} {Path(loser['path']).name} ({total} verdicts in {store})")
    return 0


def cmd_fit(args) -> int:
    store = store_dir(args.store)
    verdicts = read_verdicts(store)
    decided = [v for v in verdicts if not v.get("tie")]
    rows = fit(verdicts)
    if args.json:
        print(json.dumps({"store": str(store), "verdicts": len(verdicts), "decided": len(decided), "fit": rows}, indent=2))
        return 0
    if not decided:
        print(f"no decided verdicts in {store}")
        return 1
    print(f"{len(decided)} decided verdicts in {store}")
    table = [["metric", "winner is", "agree", "p", "median delta", "unit"]]
    for r in rows:
        table.append([r["key"], r["prefers"], f"{r['agree']}/{r['n']}", f"{r['p']:.3f}", f"{r['median_delta']:+.3f}", r["unit"]])
    print_table(table)
    if len(decided) < 6:
        print("fewer than 6 decided verdicts: no metric can reach p < 0.05 yet, treat directions as hints")
    notes = [v["note"] for v in verdicts if v.get("note")]
    if notes:
        print("notes: " + " | ".join(notes))
    return 0


def cmd_ref(args) -> int:
    store = store_dir(args.store)
    config = read_config(store)
    if args.clear:
        config.pop("ref", None)
        write_config(store, config)
    elif args.file:
        config["ref"] = load(args.file).path
        write_config(store, config)
    print(config.get("ref") or "no reference voice set")
    return 0


def default_player() -> list[str]:
    if shutil.which("afplay"):
        return ["afplay"]
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
    raise SystemExit("ears: no audio player found, pass --player")


def cmd_judge(args) -> int:
    """Blind A/B for a human at a terminal. Plays each pair in random order and records the pick."""
    player = shlex.split(args.player) if args.player else default_player()
    print("analysing files...", file=sys.stderr)
    reports = reports_for(args, args.files)
    store = store_dir(args.store)
    pairs = list(itertools.combinations(reports, 2))
    random.shuffle(pairs)
    if args.max:
        pairs = pairs[: args.max]

    def play(report: dict, quieter_by_db: float) -> None:
        cmd = list(player)
        if player[0] == "afplay" and quieter_by_db > 0:
            cmd += ["-v", f"{10 ** (-quieter_by_db / 20):.3f}"]  # loudness-match so the louder take has no edge
        subprocess.run(cmd + [report["path"]], check=False)

    for n, pair in enumerate(pairs, 1):
        first, second = random.sample(pair, 2)
        l1, l2 = first["metrics"].get("lufs"), second["metrics"].get("lufs")
        gap = (l1 - l2) if l1 is not None and l2 is not None else 0.0
        while True:
            print(f"\npair {n}/{len(pairs)}: playing 1 then 2")
            play(first, max(gap, 0.0))
            play(second, max(-gap, 0.0))
            choice = input("better? [1/2] t=tie r=replay s=skip q=quit: ").strip().lower()
            if choice != "r":
                break
        if choice == "q":
            break
        if choice not in ("1", "2", "t"):
            continue
        note = input("why? (enter to skip): ").strip()
        winner, loser = (second, first) if choice == "2" else (first, second)
        add_verdict(store, winner, loser, note=note, tie=choice == "t")
        print(f"1 = {Path(first['path']).name}, 2 = {Path(second['path']).name}")
    print(f"\n{len(read_verdicts(store))} verdicts in {store}. Run `ears fit` to see which metrics track them.")
    return 0


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--store", help="verdict store directory (default ./.ears or $EARS_STORE)")
    common.add_argument("--json", action="store_true", help="machine-readable output")
    analysis = argparse.ArgumentParser(add_help=False)
    analysis.add_argument("--ref", help="reference voice file for speaker_sim (default: the store's `ears ref`)")
    analysis.add_argument("--no-models", action="store_true", help="skip DNSMOS and speaker similarity")
    analysis.add_argument("--f0-min", type=float, default=60.0, help="pitch floor in Hz (default 60)")
    analysis.add_argument("--f0-max", type=float, default=400.0, help="pitch ceiling in Hz (default 400)")

    parser = argparse.ArgumentParser(prog="ears", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("report", parents=[common, analysis], help="metrics for one or more files, side by side")
    p.add_argument("files", nargs="+")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("diff", parents=[common, analysis], help="A/B metric deltas with notable changes flagged")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("plot", parents=[common], help="spectrogram, pitch and level PNG for one or more files")
    p.add_argument("files", nargs="+")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--start", type=float, default=0.0, help="window start in seconds")
    p.add_argument("--end", type=float, help="window end in seconds")
    p.add_argument("--tail", type=float, help="show only the last N seconds of each file")
    p.add_argument("--f0-min", type=float, default=60.0)
    p.add_argument("--f0-max", type=float, default=400.0)
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("metrics", parents=[common], help="what each metric means and its notable threshold")
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("verdict", parents=[common, analysis], help="record that a human preferred WINNER over LOSER")
    p.add_argument("winner")
    p.add_argument("loser")
    p.add_argument("--note", default="", help="the human's reason, in their words")
    p.add_argument("--tie", action="store_true", help="record as no preference")
    p.set_defaults(func=cmd_verdict)

    p = sub.add_parser("judge", parents=[common, analysis], help="interactive blind A/B listening session for a human")
    p.add_argument("files", nargs="+")
    p.add_argument("--player", help="playback command (default afplay or ffplay)")
    p.add_argument("--max", type=int, help="limit the number of pairs")
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("fit", parents=[common], help="which metrics track the recorded verdicts")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("ref", parents=[common], help="show, set or clear the store's reference voice")
    p.add_argument("file", nargs="?")
    p.add_argument("--clear", action="store_true")
    p.set_defaults(func=cmd_ref)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError, RuntimeError) as e:
        print(f"ears: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
