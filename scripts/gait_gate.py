#!/usr/bin/env python3
"""Thin wrapper over isaac_tasks/k1_velocity/scripts/gait_gate.py (reused, not copied).

Adds `evaluate(npz)` returning a dict for report writers. CLI is identical:  gait_gate.py <trace.npz>
Thresholds (S4): cadence 1.2-4.0 steps/s, stride >= 0.15 m, jerk < 0.06, plus no-swing check when feet recorded.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "isaac_tasks/k1_velocity/scripts/gait_gate.py"
_spec = importlib.util.spec_from_file_location("_k1_gait_gate_src", _SRC)
_g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_g)

analyse, main = _g.analyse, _g.main
CADENCE_MIN, CADENCE_MAX, STRIDE_MIN, JERK_MAX = _g.CADENCE_MIN, _g.CADENCE_MAX, _g.STRIDE_MIN, _g.JERK_MAX


def evaluate(npz, knee=(3, 9)) -> dict:
    res = analyse(pathlib.Path(npz), tuple(knee))
    n_bad = 0
    for r in res["per_env"]:
        bad = not (CADENCE_MIN <= r["cadence"] <= CADENCE_MAX) or r["jerk"] > JERK_MAX
        if r["stride"] == r["stride"] and r["stride"] < STRIDE_MIN:
            bad = True
        if res["has_feet"] and r.get("swing_frac", 0.0) <= 0.0:
            bad = True
        n_bad += bad
    return dict(n_env=res["n_env"], n_shuffle=int(n_bad),
                cadence=float(np.mean([r["cadence"] for r in res["per_env"]])),
                jerk=float(np.mean([r["jerk"] for r in res["per_env"]])),
                **{"pass": bool(n_bad <= res["n_env"] / 2)})


if __name__ == "__main__":
    sys.exit(main())
