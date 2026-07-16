"""
=============================================================================
load.py — LeRobot v3.0 dataset loader (local, no lerobot dependency)
=============================================================================
Loads a LeRobot-format dataset straight from disk: parquet tables for the
joint states/actions, episode metadata, and OpenCV readers for the video.

Why not just `pip install lerobot`? It drags in torch + ffmpeg + a training
stack we don't need for a data pipeline. The v3.0 on-disk format is simple
(parquet + mp4 + json), so reading it directly keeps the env light and shows
what the format actually looks like.

v3.0 layout (verified against the real files, not docs):
  meta/info.json                     -> fps, feature schema, counts
  meta/episodes/chunk-*/file-*.parquet -> one row per episode:
      dataset_from_index / dataset_to_index  = row range in the data parquet
      videos/<cam>/from_timestamp / to_timestamp = seconds into the mpv4
  data/chunk-*/file-*.parquet        -> one row per frame (all episodes concat)
  videos/<cam>/chunk-*/file-*.mp4    -> ALL episodes concatenated in one mp4

Usage: from src.load import LeRobotDataset
=============================================================================
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


class LeRobotDataset:
    """Thin read-only wrapper around a local LeRobot v3.0 directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.fps = self.info["fps"]
        self.name = self.root.name

        # camera streams are the features with dtype "video"
        self.cameras = [k for k, v in self.info["features"].items() if v.get("dtype") == "video"]

        # joint names — aloha nests them under {"motors": [...]}, svla is a flat list.
        # normalizing here so downstream code never cares
        names = self.info["features"]["observation.state"].get("names")
        self.joint_names = names["motors"] if isinstance(names, dict) else list(names)

        self.episodes = self._load_meta_parquets("meta/episodes")
        self.frames = self._load_meta_parquets("data")

    def _load_meta_parquets(self, subdir: str) -> pd.DataFrame:
        # v3 shards tables into chunk-XXX/file-XXX.parquet — glob + concat handles any count
        files = sorted((self.root / subdir).rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet under {self.root / subdir}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # ---- per-episode access ------------------------------------------------

    def episode_frames(self, episode_index: int) -> pd.DataFrame:
        """All frame rows (joint states, actions, timestamps) for one episode."""
        ep = self.episodes.loc[self.episodes["episode_index"] == episode_index].iloc[0]
        # row range comes from episode meta — faster + safer than filtering 25k rows
        return self.frames.iloc[int(ep["dataset_from_index"]) : int(ep["dataset_to_index"])]

    def state_matrix(self, episode_index: int) -> np.ndarray:
        """Joint states as a (frames, joints) float array — what audit code wants."""
        return np.stack(self.episode_frames(episode_index)["observation.state"].to_numpy())

    def video_frames(self, episode_index: int, camera: str, stride: int = 1):
        """
        Yield (frame_index_in_episode, BGR image) for one episode of one camera.

        All episodes live in ONE concatenated mp4, so we seek to the episode's
        start frame and read its length. Seeking by frame number (not msec)
        because msec seeking lands on keyframes and silently misaligns.
        """
        ep = self.episodes.loc[self.episodes["episode_index"] == episode_index].iloc[0]
        chunk = int(ep[f"videos/{camera}/chunk_index"])
        file = int(ep[f"videos/{camera}/file_index"])
        path = self.root / "videos" / camera / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"

        start = int(round(float(ep[f"videos/{camera}/from_timestamp"]) * self.fps))
        n = int(ep["length"])

        cap = cv2.VideoCapture(str(path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for i in range(n):
                ok, img = cap.read()
                if not ok:  # ran off the end of the file — caller should treat as dropout
                    break
                if i % stride == 0:
                    yield i, img
        finally:
            cap.release()

    def __repr__(self):
        return (f"LeRobotDataset({self.name}: {len(self.episodes)} eps, "
                f"{len(self.frames)} frames, {self.fps} fps, cams={self.cameras})")
