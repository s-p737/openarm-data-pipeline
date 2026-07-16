# OpenArm 2.0 — Egocentric Data & Teleoperation Pipeline

Take-home submission. Tasks completed: **1, 2, 3, 4, and 5 (bonus, VLA)**, where Tasks 1 and 3 have working code against real LeRobot datasets, and Tasks 2/4/5 are design docs.

| task | where | what |
|---|---|---|
| 1. Exploration & quality audit | [`notebooks/01_exploration.ipynb`](notebooks/01_exploration.ipynb) + `src/audit.py`, `src/video_quality.py` | profiling, anomaly detection, and findings on both modalities |
| 2. Labeling design | [`docs/02_labeling_design.md`](docs/02_labeling_design.md) | 3-level schema, tooling, inter-annotator agreement, temporal alignment |
| 3. Curation pipeline | `src/curate.py` (+ `src/load.py`) | filter → clean → save curated subset + decision report |
| 4. Policy evaluation design | [`docs/04_eval_design.md`](docs/04_eval_design.md) | metrics, rollout counts, sim-to-real diagnosis, ego success detector |
| 5. VLA adaptation (bonus) | [`docs/05_vla_adaptation.md`](docs/05_vla_adaptation.md) | OpenVLA fine-tuning: format, hyperparameters, egocentric failure modes |

## Data

Two public LeRobot datasets, chosen so both modalities are actually exercised:

- **`lerobot/aloha_sim_insertion_human`**: the dataset named in the prompt.
  50 sim episodes, 14 joints, 50 fps, one top camera. It has no wrist
  camera, so for the egocentric half I added:
- **`lerobot/svla_so100_pickplace`**: 50 SO100 episodes, 30 fps,
  **top + wrist cameras**. Real hardware also means real defects (quantized
  encoders, operator pauses), which made the audit findings much less
  hypothetical.

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# pull the datasets (~550 MB total)
python -c "
from huggingface_hub import snapshot_download
for r in ['lerobot/aloha_sim_insertion_human', 'lerobot/svla_so100_pickplace']:
    snapshot_download(r, repo_type='dataset', local_dir='data_raw/' + r.split('/')[1])"

# run the curation pipeline (Task 3)
python -m src.curate data_raw/aloha_sim_insertion_human
python -m src.curate data_raw/svla_so100_pickplace          # includes wrist-cam checks
python -m src.curate data_raw/svla_so100_pickplace --dry-run  # decisions only, no writes

# standalone audits (Task 1, also embedded in the notebook)
python -m src.audit data_raw/svla_so100_pickplace
python -m src.video_quality data_raw/svla_so100_pickplace wrist
```

Outputs land in `out/<dataset>/`: `curated_frames.parquet` (cleaned joint
data, raw preserved alongside), `bad_frame_masks.json` (frame-level video
defects), `curation_report.json` (every keep/drop decision with reasons and
the exact config that produced it).

## Architecture & design decisions

```
data_raw/ (LeRobot v3.0)                       out/
  ├─ joints ──► audit.py ───┐                   ├─ curated_frames.parquet
  │                         ├──► curate.py ───► ├─ bad_frame_masks.json
  └─ videos ──► video_quality.py ┘              └─ curation_report.json
```

**No `lerobot` dependency.** The library drags in torch + a training stack; a
data pipeline needs none of it. `src/load.py` reads the v3.0 format directly
(parquet + concatenated mp4 + json metadata) in ~100 lines, verified against
the real files. Fewer deps, and it forced me to learn and understand the formatting myself.

**Audit checks are pure functions, used twice.** The notebook explores with the code the pipeline filters with, ensuring that the analysis can't drift from the production behavior.

**Episodes get dropped + frames get masked.** Deleting individual video frames
breaks the 1:1 temporal alignment with the joint stream, so frame-level
defects (blur, clipping) are recorded as masks for the training dataloader to
skip, and only episode-level defects cause deletions. Videos are never
re-encoded; curated data points back into the source mp4s via timestamps.

**Thresholds are not absolute and are physical/robust** The two datasets disagree
about everything absolute (radians vs degrees, float-jitter vs 12-bit
quantized encoders, 50 vs 30 fps). Jump detection uses MAD + a physical floor
("no joint sweeps its full range in <0.25 s"); blur thresholds are relative to
each stream's own median. Both of these were bugs at first as v1 of the jump check
flagged 500 false positives per episode, and v1 of the stall check flagged
every svla episode because quantized encoders legitimately repeat values. The
notebook documents both iterations.

**Deterministic and reproducible.** Identical inputs produce byte-identical parquet outputs (verified by hash). Each report embeds the config that generated it, so any curated set can be traced back to its exact settings.

### What the audits found

- **aloha**: 8/50 episodes contain single-frame teleports (one joint moves an
  implausible amount in 20 ms) → dropped. Zero NaNs / timing gaps.
- **svla**: median episode is ~19% fully-stalled (operator pauses — 
  indistinguishable from bus dropout using positions alone); 4 episodes exceed
  30% dead time (dropped). Wrist-cam blur is ~10× worse than the top cam and
  bursts during fast reaches (up to 9% of frames) → masked, episodes kept.

## Teleop vs egocentric: the key trade-offs
Teleop and egocentric video fail you in almost opposite ways, and that difference shapes a lot of things downstream. With joint states, a defect is something that is physically impossible, like a joint value outside its actual limits, a jump no actuator could have made. That means you can ground your thresholds in physical limits and statistics, and you can afford to be strict about it, because a bad state is just wrong. Egocentric video doesn't give you that certainty. A blurry or occluded frame is perceptually useless without being physically impossible, so your thresholds only make sense relative to the statistics of that particular stream rather than any fixed standard. This difference in what even counts as a defect changes how you're allowed to filter. A bad joint state usually means dropping the whole episode, while a bad frame calls for a mask, and you only escalate to dropping the episode once more than about 20% of it is bad. It also changes how aggressive you can be. Joint states can be handled strictly, since a wrong value is unambiguously wrong. Video has to be handled conservatively, since even a blurry frame is still carrying information you don't want to throw away. And each modality has a trap; for joint states, it's the unit and encoder assumptions that corrupt your thresholds without you noticing. For video, it's that filtering out blur ends up selecting for slow demonstrations, so the policy you train on that data learns to be slow too.

## What I'd do next (more time / hardware access)

1. **Task-space checks.** Right now everything is validated per joint. Running
   forward kinematics to audit the actual end-effector path would catch
   near-collisions and workspace exits that are invisible in joint space.
2. **Occlusion detection on the wrist cam.** Blur and exposure are cheap solved
   problems. The remaining quality axis is whether the gripper is blocking the
   view, and it matters most right at grasp time, so a gripper-occupancy
   segmentation check is the obvious next filter.
3. **Close the loop.** Train the Task 4 egocentric success detector on the
   Task 2 labels and run it on incoming episodes, so bad data gets flagged
   during collection rather than weeks later in curation.
4. **On real hardware:** bus-level comms logging, since that's the only reliable
   way to distinguish sensor dropout from an operator pausing, plus
   camera-to-joint clock calibration. This pipeline trusts LeRobot's
   timestamps. A real collection rig has to earn them.
   
## Assumptions

- Success/task labels don't exist in these datasets, so curation is purely
  quality-based; with labels, outcome-aware filtering comes first.
- Curated output stays in parquet + source-video references instead of a
  re-packaged LeRobot dataset (faster to iterate; conversion is mechanical).
- Wrist cam ≈ egocentric. True for OpenArm's setup as described; a head-mounted
  rig would add gaze/parallax questions this design doesn't cover.
