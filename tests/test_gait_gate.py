"""scripts/gait_gate.py on synthetic traces: a walk passes, a chatter fails."""
import runpy
from pathlib import Path

import numpy as np

GATE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/gait_gate.py"))


def _trace(tmp_path, knee_hz, speed, noise=0.0):
    steps, envs, joints = 500, 4, 12
    t = np.arange(steps) * GATE["DT"]
    rng = np.random.default_rng(0)
    actions = np.zeros((steps, envs, joints))
    actions[:, :, 3] = 0.4 * np.sin(2 * np.pi * knee_hz * t)[:, None]
    actions[:, :, 9] = 0.4 * np.sin(2 * np.pi * knee_hz * t + np.pi)[:, None]
    actions += noise * rng.standard_normal(actions.shape)
    vel = np.zeros((steps, envs, 3))
    vel[..., 0] = speed
    path = tmp_path / "trace.npz"
    np.savez(path, actions=actions, root_lin_vel=vel)
    return path


def test_walk_passes(tmp_path):
    # 1 Hz knee cycle = 2 steps/s; 0.5 m/s -> 0.25 m stride; smooth
    res = GATE["evaluate"](_trace(tmp_path, 1.0, 0.5))
    assert res["pass"] and res["n_shuffle"] == 0, res
    assert abs(res["cadence"] - 2.0) < 0.1, res


def test_chatter_fails(tmp_path):
    # 4 Hz knee cycle = 8 steps/s on 6 cm strides plus jitter: the failure the gate exists for
    res = GATE["evaluate"](_trace(tmp_path, 4.0, 0.5, noise=0.05))
    assert not res["pass"] and res["n_shuffle"] == res["n_env"], res
