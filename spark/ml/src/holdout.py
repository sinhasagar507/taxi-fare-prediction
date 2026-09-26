"""Phase 5's one score on the sealed sets (modeling plan §4a, §6 "HOLDOUT").

§4a fixes the rule — the holdout is scored exactly once, after the champion is
chosen on CV alone — and calls it "the part no code enforces". This module
enforces it:

- `guard_once` refuses to go on when a score file already exists;
- `write_score_once` creates that file with exclusive-create mode, so a second
  write fails even if the guard was skipped;
- `score_sealed` fits the pipeline once and returns metrics only, never a
  prediction, so nothing row-level can reach a log.

The score file lives in the gitignored results directory, so deleting it resets
the guard. The committed docs are the permanent record of the one score.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from spark.ml.src.evaluate import compute_metrics


class SealedScoreExists(RuntimeError):
    """The sealed sets were already scored; a second score is refused."""


def guard_once(path: str | Path) -> None:
    """Raise if the one score has already been written to `path`."""
    if Path(path).exists():
        raise SealedScoreExists(
            f"{path} exists — the sealed sets were already scored once (§4a). "
            "Read that file; do not score again."
        )


def write_score_once(result: dict, path: str | Path) -> Path:
    """Write the one score, refusing if the file already exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x") as fh:
            json.dump(result, fh, indent=2)
    except FileExistsError as exc:
        raise SealedScoreExists(f"{path} exists — refusing a second score") from exc
    return path


def _metrics(y, pred) -> dict:
    return {"rows": int(len(y)), **{k: float(v) for k, v in compute_metrics(y, pred).items()}}


def score_sealed(pipeline, X_train, y_train, sealed: dict, groups: dict | None = None) -> dict:
    """Fit `pipeline` once on the train split, then score each sealed set.

    `sealed` maps a set name to its (X, y). `groups` optionally maps a set name
    to a label Series aligned with that set's rows (e.g. the pickup month); its
    metrics are then also reported per label. Returns numbers only.
    """
    started = time.time()
    pipeline.fit(X_train, y_train)
    fit_time = time.time() - started

    out = {"fit_time_s": float(fit_time), "sets": {}}
    for name, (X, y) in sealed.items():
        pred = np.asarray(pipeline.predict(X))
        entry = _metrics(y, pred)
        if groups and name in groups:
            labels = np.asarray(groups[name])
            y_arr = np.asarray(y)
            entry["by_group"] = {
                str(label): _metrics(y_arr[labels == label], pred[labels == label])
                for label in sorted(set(labels))
            }
        out["sets"][name] = entry
    return out
