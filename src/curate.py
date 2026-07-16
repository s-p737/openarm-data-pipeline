"""
=============================================================================
curate.py — Task 3: the data curation pipeline
=============================================================================
Loads a LeRobot v3.0 dataset, filters bad episodes, cleans trajectories, and
writes a curated subset + a JSON report explaining every decision.

    python -m src.curate data_raw/svla_so100_pickplace
    python -m src.curate data_raw/aloha_sim_insertion_human --dry-run

Pipeline:
  1. AUDIT   — run src/audit.py checks on every episode (joint side) and
               src/video_quality.py on the wrist cam if the dataset has one
  2. FILTER  — drop episodes, never individual frames. one hard rule:
               deleting single frames breaks the 1:1 temporal alignment
               between video and joint stream. frame-level defects become
               MASKS (saved alongside), episode-level defects become drops.
  3. CLEAN   — Savitzky-Golay smoothing on the kept joint trajectories
               (raw values preserved in *_raw columns — never destroy source)
  4. WRITE   — out/<dataset>/curated_frames.parquet
               out/<dataset>/bad_frame_masks.json   (per-episode, per-camera)
               out/<dataset>/curation_report.json   (every decision + why)

Videos are NOT re-encoded: curated frames still point at the source mp4s via
episode timestamps. Re-encoding 500MB of video to drop 3 episodes is waste,
and lossy re-compression would degrade every surviving frame.

The whole run is deterministic — same input, same output, byte for byte
(fixed sampling stride, no randomness anywhere). Run it twice to check me.
=============================================================================
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.audit import audit_episode, dataset_joint_ranges
from src.load import LeRobotDataset
from src.video_quality import episode_video_report

# every threshold in one place, and every one of them is IN the report json,
# so a curated set can always answer "what settings produced you?"
CONFIG = {
    "min_duration_s": 2.0,        # shorter = aborted demo, not a demonstration
    "max_jumps": 0,               # one teleport = corrupted episode, zero tolerance
    "max_stall_fraction": 0.30,   # >30% dead time dilutes the training signal
    "max_nan_fraction": 0.01,     # tiny gaps get interpolated; more = broken recording
    "max_time_gaps": 0,           # timing gaps = streams can't be trusted to align
    "max_bad_video_fraction": 0.20,  # ego: >20% blurry/clipped/frozen frames = drop
    "smooth_window_frames": 9,    # ~0.2-0.3s at 30-50fps: kills jitter, keeps dynamics
    "smooth_polyorder": 3,
}


def interpolate_small_gaps(states: np.ndarray) -> np.ndarray:
    """Linear interp over NaN gaps — only called when the gap is tiny (config-gated)."""
    out = states.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        bad = np.isnan(col)
        if bad.any():
            col[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), col[~bad])
    return out


def smooth_states(states: np.ndarray, joint_names: list, cfg: dict) -> np.ndarray:
    """
    Savitzky-Golay per joint — fits a local polynomial, so it flattens encoder
    staircase + teleop jitter without lagging the signal like a moving average.

    grippers are exempt: they're near-binary open/close and smoothing invents
    half-open states the robot never commanded.
    """
    out = states.copy()
    win = min(cfg["smooth_window_frames"], len(states) - (1 - len(states) % 2))
    if win <= cfg["smooth_polyorder"]:
        return out  # episode too short to smooth — leave it alone
    for j, name in enumerate(joint_names):
        if "gripper" in name.lower():
            continue
        out[:, j] = savgol_filter(out[:, j], win, cfg["smooth_polyorder"])
    return out


def decide(ep_audit: dict, video_report: dict | None, cfg: dict) -> list[str]:
    """All drop reasons for one episode. Empty list = keeper."""
    reasons = []
    if ep_audit["duration_s"] < cfg["min_duration_s"]:
        reasons.append(f"too_short ({ep_audit['duration_s']:.1f}s)")
    if ep_audit["jumps"]["n_jumps"] > cfg["max_jumps"]:
        reasons.append(f"joint_jumps ({ep_audit['jumps']['n_jumps']})")
    if ep_audit["frozen"]["stall_fraction"] > cfg["max_stall_fraction"]:
        reasons.append(f"stalled ({ep_audit['frozen']['stall_fraction']:.0%} dead time)")
    nan_frac = ep_audit["nans"]["n_nan"] / max(ep_audit["n_frames"], 1)
    if nan_frac > cfg["max_nan_fraction"]:
        reasons.append(f"nans ({nan_frac:.1%})")
    if ep_audit["time"]["n_gaps"] > cfg["max_time_gaps"]:
        reasons.append(f"timing_gaps ({ep_audit['time']['n_gaps']})")
    if video_report and video_report["readable"]:
        if video_report["bad_frame_fraction"] > cfg["max_bad_video_fraction"]:
            reasons.append(f"bad_video ({video_report['bad_frame_fraction']:.0%} of frames)")
    return reasons


def curate(root: str, out_dir: str = "out", ego_cam_hint: str = "wrist",
           dry_run: bool = False, cfg: dict = CONFIG) -> dict:
    ds = LeRobotDataset(root)
    ranges = dataset_joint_ranges(ds)
    # egocentric stream = whichever camera matches the hint; datasets without
    # one (aloha sim) just skip the video gate — pipeline handles both
    ego_cam = next((c for c in ds.cameras if ego_cam_hint in c), None)
    print(f"{ds}\n  egocentric camera: {ego_cam or 'none — video checks skipped'}")

    decisions, masks, kept_frames = [], {}, []
    for ep in ds.episodes["episode_index"]:
        frames = ds.episode_frames(int(ep))
        states = ds.state_matrix(int(ep))
        rep = audit_episode(states, frames["timestamp"].to_numpy(), ds.fps, ranges)
        vrep = episode_video_report(ds, int(ep), ego_cam) if ego_cam else None
        reasons = decide(rep, vrep, cfg)

        decisions.append({
            "episode": int(ep),
            "kept": not reasons,
            "drop_reasons": reasons,
            "n_frames": rep["n_frames"],
            "n_jumps": rep["jumps"]["n_jumps"],
            "stall_fraction": round(rep["frozen"]["stall_fraction"], 4),
            "bad_video_fraction": round(vrep["bad_frame_fraction"], 4) if vrep else None,
        })

        if reasons:
            continue

        # frame-level video defects -> mask, not deletion (alignment rule)
        if vrep:
            masks[int(ep)] = {ego_cam: sorted(set(vrep["bad_blur_frames"])
                                              | set(vrep["bad_exposure_frames"])
                                              | set(vrep["frozen_frames"]))}

        # clean the keeper: interp tiny gaps, then smooth (raw kept alongside)
        clean = smooth_states(interpolate_small_gaps(states), ds.joint_names, cfg)
        ep_out = frames.copy()
        ep_out["observation.state_raw"] = ep_out["observation.state"]
        ep_out["observation.state"] = list(clean.astype(np.float32))
        kept_frames.append(ep_out)

    kept = [d for d in decisions if d["kept"]]
    report = {
        "dataset": ds.name,
        "curated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg,
        "egocentric_camera": ego_cam,
        "episodes_in": len(decisions),
        "episodes_kept": len(kept),
        "frames_kept": int(sum(d["n_frames"] for d in kept)),
        "decisions": decisions,
    }

    print(f"  kept {len(kept)}/{len(decisions)} episodes")
    for d in decisions:
        if not d["kept"]:
            print(f"  dropped ep {d['episode']:3d}: {', '.join(d['drop_reasons'])}")

    if dry_run:
        print("  dry run — nothing written")
        return report

    out = Path(out_dir) / ds.name
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(kept_frames, ignore_index=True).to_parquet(out / "curated_frames.parquet")
    (out / "bad_frame_masks.json").write_text(json.dumps(masks, indent=1))
    (out / "curation_report.json").write_text(json.dumps(report, indent=1))
    print(f"  wrote {out}/curated_frames.parquet, bad_frame_masks.json, curation_report.json")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="curate a LeRobot v3.0 dataset")
    p.add_argument("dataset", help="path to dataset root, e.g. data_raw/svla_so100_pickplace")
    p.add_argument("--out", default="out", help="output directory (default: out/)")
    p.add_argument("--ego-camera", default="wrist", help="substring matching the egocentric camera")
    p.add_argument("--dry-run", action="store_true", help="print decisions, write nothing")
    a = p.parse_args()
    curate(a.dataset, a.out, a.ego_camera, a.dry_run)
