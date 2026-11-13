# agent-ears

Lets a coding agent hear speech audio well enough to iterate on it.

An agent refining generated voice audio cannot listen. `ears` gives it two things:

1. **Measurements.** A fixed set of acoustic metrics per file, A/B deltas with notable changes flagged, and a spectrogram PNG the agent can read.
2. **Direction.** A log of a human's listening verdicts, and a fit that shows which metrics track those verdicts.

Measurements say how two takes differ. Verdicts say which difference the human wants. Everything runs locally.

## Install

Requires Python 3.11 to 3.13 and [uv](https://docs.astral.sh/uv/).

```sh
uv tool install .        # puts `ears` on PATH
# or, from a checkout:
uv sync && uv run ears --help
```

The first `report` downloads two ONNX models (28 MB total, sha256-pinned) to `~/.cache/agent-ears/models`. Set `EARS_CACHE` to move the cache. Pass `--no-models` to skip them.

## Commands

| Command | Purpose |
|---|---|
| `ears report FILE...` | Metrics for one or more files, side by side. |
| `ears diff A B` | Metric deltas, `*` on notable changes, plus the direction the human's picks favor. |
| `ears plot FILE... -o out.png` | Mel spectrogram, pitch and level on a shared time axis. `--tail 3` shows the last 3 seconds. |
| `ears metrics` | What each metric means and its notable threshold. |
| `ears verdict WINNER LOSER --note "..."` | Record that a human preferred one file. `--tie` records no preference. |
| `ears judge FILE...` | Interactive blind A/B session for a human. Plays every pair in random order, loudness-matched, and records the picks. |
| `ears fit` | Which metrics track the recorded verdicts. |
| `ears ref [FILE]` | Set the reference voice used for `speaker_sim`. |

Every command takes `--json`. Exit code is 0 on success and 2 on error.

## Metrics

Run `ears metrics` for the full list. Groups:

- **Level:** loudness (LUFS), peak, clipping, level variation.
- **Timing:** leading and trailing silence, abrupt ending, pause map, estimated syllable rate.
- **Pitch:** median, range and variation in semitones, final contour slope.
- **Voice quality:** jitter, shimmer, harmonics-to-noise ratio, cepstral peak prominence.
- **Timbre:** spectral centroid, 1-5 kHz to low-band ratio, high-frequency ratio.
- **Model-backed:** DNSMOS predicted quality, speaker-embedding similarity to a reference voice.

All files are analysed as mono at 24 kHz so numbers compare across files.

## Verdict store

Verdicts live in `./.ears/verdicts.jsonl` (override with `--store` or `EARS_STORE`). Each verdict snapshots both files' metrics, so `ears fit` still works after the audio is deleted. Keep one store per listening goal, since preferences for one voice may not transfer to another.

`ears fit` runs a sign test per metric: in how many decided pairs did the winner have the higher value. Fewer than 6 decided pairs cannot reach p < 0.05, so early directions are hints.

## For agents

1. After generating a new take, run `ears diff PREVIOUS NEW`. Report the `*` rows to the human in plain words.
2. To inspect a specific moment (an ending, a glitch, a pause), run `ears plot` with `--tail` or `--start/--end` and read the PNG.
3. When the human states a preference in chat, record it with `ears verdict WINNER LOSER --note "their words"`. Never record a verdict the human did not give.
4. When several takes need a human ear, ask the human to run `ears judge FILES...` in their terminal.
5. Before choosing the next change, run `ears fit` and aim at the metrics the picks agree on.

## Limits

- The metrics describe sound. They do not check words. Use ASR for script accuracy.
- Syllable rate is estimated from intensity peaks, not from a transcript.
- DNSMOS was trained to rate noise-suppression output. Treat it as a rough quality proxy for TTS.
- The notable thresholds are rough audibility guesses.
- Jitter, shimmer and HNR are computed over running speech, so compare them between takes of the same script, not against clinical norms.
- A metric that tracks the picks is correlated with them. It may not be what the human is hearing.

## Development

```sh
uv sync
uv run pytest
```
