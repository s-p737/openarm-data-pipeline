"""
=============================================================================
audit.py — Task 1: quality checks for teleoperation joint-state data
=============================================================================
Every check returns plain dicts (JSON-able) so the same functions feed the
exploration notebook, the curation pipeline, and the audit report. No side
effects, no printing — pure functions over one episode's state matrix.

The checks:
  1. jump_anomalies    — physically implausible frame-to-frame jumps
  2. frozen_runs       — sensor dropout (values stuck while robot should move)
  3. nan_check         — missing values
  4. timestamp_gaps    — recording glitches (dt far from 1/fps)
  5. episode profile   — length, per-joint range (for the length histogram etc.)

Design choice worth defending: thresholds are ROBUST + RELATIVE, not absolute.
aloha joints are in radians (~±1), svla in degrees (~0–180) — any fixed
threshold breaks on one of them. So jumps are flagged against the dataset's
own delta distribution (median + MAD), which is unit-agnostic for free.

Run standalone: python -m src.audit data_raw/aloha_sim_insertion_human
=============================================================================
"""

import sys

import numpy as np

# how many robust stds a frame-to-frame delta must be to count as a jump.
# 12 is deliberately loose — fast intentional motion is normal, we only want
# physically impossible teleports. tightening this biases toward slow demos
JUMP_SIGMA = 12.0

# physical floor for the jump check: nothing sweeps its full joint range in
# under this many seconds, so per-frame delta must ALSO exceed
# range * (1 / (MIN_SWEEP_SECONDS * fps)) to count as a teleport.
# without this floor, a joint that idles most of the episode has ~zero MAD
# and every real movement flags (first version flagged 500 "jumps"/episode)
MIN_SWEEP_SECONDS = 0.25

# a joint frozen for >= this many SECONDS while other joints move = suspicious.
# short freezes are just the arm holding still, which is fine
FROZEN_SECONDS = 1.0

# dt more than 50% off from 1/fps counts as a recording gap
DT_TOLERANCE = 0.5


def robust_std(x: np.ndarray) -> float:
    """MAD-based std estimate — one bad spike can't inflate it like np.std can."""
    mad = np.median(np.abs(x - np.median(x)))
    return float(mad * 1.4826) or 1e-9  # never return 0, we divide by this


def jump_anomalies(states: np.ndarray, fps: float, joint_ranges: np.ndarray,
                   sigma: float = JUMP_SIGMA) -> dict:
    """
    Flag frame-to-frame deltas that are (a) way outside the episode's own
    motion scale AND (b) above a physical speed limit.

    joint_ranges = dataset-wide (max - min) per joint. the physical floor
    makes the check unit-agnostic without magic numbers: radians, degrees,
    whatever — "full range in <0.25 s" is a teleport in any unit.
    """
    deltas = np.diff(states, axis=0)  # (frames-1, joints)
    floor = joint_ranges / (MIN_SWEEP_SECONDS * fps)  # max plausible per-frame delta
    flags = []
    for j in range(deltas.shape[1]):
        thresh = max(sigma * robust_std(deltas[:, j]), floor[j])
        bad = np.where(np.abs(deltas[:, j]) > thresh)[0]
        for f in bad:
            flags.append({"frame": int(f + 1), "joint": j, "delta": float(deltas[f, j])})
    return {"n_jumps": len(flags), "jumps": flags[:20]}  # cap the list, report the count


def frozen_runs(states: np.ndarray, fps: float, min_seconds: float = FROZEN_SECONDS) -> dict:
    """
    Dropout detector — and the check that taught me you have to calibrate to
    the SENSOR, not just the units.

    v1 flagged any joint bit-identical for >1s while others moved. worked on
    the aloha sim (float states jitter at 1e-8, exact repeats = dropout) but
    fired on 50/50 svla episodes: the SO100's encoder is 12-bit quantized
    (min step 0.0879 deg = 360/4096, 43% of frames have zero delta on a
    joint). bit-identical repeats are NORMAL there.

    so: auto-detect quantization from the data. quantized encoders only flag
    a FULL-state stall (every joint frozen at once); continuous encoders keep
    the per-joint check.

    v2 lesson (checked the flagged episodes): on svla the median episode is
    ~19% fully-stalled with ~1.5s longest runs — that's the OPERATOR PAUSING
    mid-teleop, not dropout. positions alone can't tell a pause from a bus
    stall; you'd need comms-level logging on the real robot. so the number
    that matters for curation is stall_fraction (dead time dilutes training
    signal), not "did a freeze ever happen".
    """
    min_run = int(min_seconds * fps)
    deltas = np.diff(states, axis=0)
    # median-over-joints fraction of exactly-zero deltas: ~0 for float sim
    # states, ~0.4 for quantized servo encoders
    zero_frac = float(np.median((deltas == 0.0).mean(axis=0)))
    quantized = zero_frac > 0.10

    def runs_of(same: np.ndarray, joint) -> list:
        found, run = [], 0
        for f in range(len(same)):
            run = run + 1 if same[f] else 0
            if run >= min_run:
                found.append({"joint": joint, "end_frame": int(f + 1), "run_frames": run})
                run = 0  # report each long run once
        return found

    full_stall = (deltas == 0.0).all(axis=1)  # nothing on the robot changed
    worst = []
    if quantized:
        worst += runs_of(full_stall, joint="ALL")
    else:
        moving = np.abs(deltas).max(axis=1) > 0  # a still robot isn't a dropout
        for j in range(states.shape[1]):
            worst += [r for r in runs_of(deltas[:, j] == 0.0, joint=j)
                      if moving[r["end_frame"] - 1]]

    # longest contiguous full stall, in seconds
    longest = cur = 0
    for v in full_stall:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)

    return {
        "n_frozen_runs": len(worst),
        "quantized_encoder": quantized,
        "stall_fraction": float(full_stall.mean()),  # the metric curation actually uses
        "longest_stall_s": float(longest / fps),
        "runs": worst[:10],
    }


def nan_check(states: np.ndarray) -> dict:
    return {"n_nan": int(np.isnan(states).sum())}


def timestamp_gaps(timestamps: np.ndarray, fps: float, tol: float = DT_TOLERANCE) -> dict:
    """Recording glitches: dt between frames straying far from the nominal 1/fps."""
    dt = np.diff(timestamps)
    nominal = 1.0 / fps
    bad = np.where(np.abs(dt - nominal) > tol * nominal)[0]
    return {"n_gaps": len(bad), "worst_dt": float(dt.max()) if len(dt) else None}


def audit_episode(states: np.ndarray, timestamps: np.ndarray, fps: float,
                  joint_ranges: np.ndarray) -> dict:
    """Run every check on one episode; the curation pipeline consumes this dict."""
    return {
        "n_frames": int(states.shape[0]),
        "duration_s": float(states.shape[0] / fps),
        "jumps": jump_anomalies(states, fps, joint_ranges),
        "frozen": frozen_runs(states, fps),
        "nans": nan_check(states),
        "time": timestamp_gaps(timestamps, fps),
    }


def dataset_joint_ranges(ds) -> np.ndarray:
    """Dataset-wide per-joint (max - min), the scale for the physical jump floor."""
    all_states = np.stack(ds.frames["observation.state"].to_numpy())
    return all_states.max(axis=0) - all_states.min(axis=0)


if __name__ == "__main__":
    # quick CLI smoke test: audit every episode, print anything suspicious
    from src.load import LeRobotDataset

    ds = LeRobotDataset(sys.argv[1])
    print(ds)
    ranges = dataset_joint_ranges(ds)
    flagged = 0
    for ep_idx in ds.episodes["episode_index"]:
        frames = ds.episode_frames(ep_idx)
        rep = audit_episode(ds.state_matrix(ep_idx), frames["timestamp"].to_numpy(), ds.fps, ranges)
        # a stalled episode is an issue when a big chunk of it is dead time,
        # not when the operator ever paused (that's every real teleop demo)
        stalled = rep["frozen"]["stall_fraction"] > 0.3
        issues = rep["jumps"]["n_jumps"] + int(stalled) + rep["nans"]["n_nan"] + rep["time"]["n_gaps"]
        if issues:
            flagged += 1
            print(f"ep {ep_idx:3d}: {rep['n_frames']} frames | jumps={rep['jumps']['n_jumps']} "
                  f"stall={rep['frozen']['stall_fraction']:.0%} nans={rep['nans']['n_nan']} gaps={rep['time']['n_gaps']}")
    print(f"\n{flagged}/{len(ds.episodes)} episodes with at least one flag")
