"""
=============================================================================
video_quality.py — Task 1/3: quality checks for egocentric (wrist cam) video
=============================================================================
Egocentric video fails differently than joint states. The camera is BOLTED TO
THE MOVING ARM, so:
  - motion blur is constant (every fast reach smears the image)
  - the gripper/object occludes the view at the most important moments
  - exposure swings as the camera points at the light vs. the table

Per-frame metrics (all cheap, all classical CV — no model needed):
  blur      — variance of the Laplacian on grayscale. sharp edges -> high
              variance; blur kills edges -> low. the standard trick.
  exposure  — mean brightness + fraction of pixels crushed to black/white
  freeze    — near-zero diff from previous frame = duplicated/frozen frame
              (video-side dropout, the twin of audit.py's frozen_runs)

Design choice worth defending: the blur threshold is RELATIVE to the
episode's own median, not absolute. Laplacian variance depends on scene
texture — a plain table scores "blurrier" than a cluttered one at the same
sharpness. And we stay conservative on purpose: blur during fast motion is
real signal, and filtering it hard would bias the dataset toward slow demos.

Run standalone: python -m src.video_quality data_raw/svla_so100_pickplace wrist
=============================================================================
"""

import sys

import cv2
import numpy as np

# frame is "bad-blurry" if its laplacian var drops below this fraction of the
# episode median — loose on purpose (see header)
BLUR_REL_THRESHOLD = 0.30

# exposure: >30% of pixels crushed to either end = frame is clipped
CLIP_FRACTION = 0.30
DARK, BRIGHT = 15, 240  # 8-bit crush points

# mean abs diff between consecutive sampled frames below this = frozen video
FREEZE_DIFF = 0.5

# sample every Nth frame — quality stats don't need all 30/50fps,
# and this keeps a full-dataset pass under a minute
DEFAULT_STRIDE = 5


def frame_metrics(img_bgr: np.ndarray) -> dict:
    """All per-frame quality numbers from one BGR frame."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return {
        "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "dark_frac": float((gray < DARK).mean()),
        "bright_frac": float((gray > BRIGHT).mean()),
    }


def episode_video_report(ds, episode_index: int, camera: str,
                         stride: int = DEFAULT_STRIDE) -> dict:
    """
    Sample one episode's video and summarize quality. Returns a JSON-able dict
    with per-frame metrics + which sampled frames failed which check.
    `ds` is a src.load.LeRobotDataset.
    """
    rows, prev_gray = [], None
    for i, img in ds.video_frames(episode_index, camera, stride=stride):
        m = frame_metrics(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # frozen-frame check: compare against previous SAMPLED frame
        m["diff"] = float(np.abs(gray.astype(np.int16) - prev_gray).mean()) if prev_gray is not None else None
        m["frame"] = i
        prev_gray = gray.astype(np.int16)
        rows.append(m)

    if not rows:
        return {"n_sampled": 0, "readable": False}

    blurs = np.array([r["blur"] for r in rows])
    blur_floor = np.median(blurs) * BLUR_REL_THRESHOLD

    bad_blur = [r["frame"] for r in rows if r["blur"] < blur_floor]
    bad_exposure = [r["frame"] for r in rows
                    if r["dark_frac"] > CLIP_FRACTION or r["bright_frac"] > CLIP_FRACTION]
    frozen = [r["frame"] for r in rows if r["diff"] is not None and r["diff"] < FREEZE_DIFF]

    bad = sorted(set(bad_blur) | set(bad_exposure) | set(frozen))
    return {
        "n_sampled": len(rows),
        "readable": True,
        "blur_median": float(np.median(blurs)),
        "blur_floor": float(blur_floor),
        "brightness_mean": float(np.mean([r["brightness"] for r in rows])),
        "bad_blur_frames": bad_blur,
        "bad_exposure_frames": bad_exposure,
        "frozen_frames": frozen,
        "bad_frame_fraction": len(bad) / len(rows),
        "frames": rows,  # full per-frame table for the notebook plots
    }


if __name__ == "__main__":
    # smoke test: report bad-frame fraction for every episode of one camera
    from src.load import LeRobotDataset

    root, cam_short = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "wrist")
    ds = LeRobotDataset(root)
    camera = next(c for c in ds.cameras if cam_short in c)
    print(ds, "| camera:", camera)
    for ep_idx in ds.episodes["episode_index"]:
        r = episode_video_report(ds, ep_idx, camera)
        flag = "  <-- check me" if r["bad_frame_fraction"] > 0.2 else ""
        print(f"ep {ep_idx:3d}: sampled={r['n_sampled']:3d} blur_med={r['blur_median']:7.1f} "
              f"bad={r['bad_frame_fraction']:.0%}{flag}")
