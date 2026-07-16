# OpenArm 2.0 — Egocentric Data & Teleoperation Pipeline

Take-home submission. Tasks completed: **1, 2, 3, 4, and 5 (bonus, VLA)** —
Tasks 1 and 3 as working code against real LeRobot datasets, Tasks 2/4/5 as
design docs.

| task | where | what |
|---|---|---|
| 1. Exploration & quality audit | [`notebooks/01_exploration.ipynb`](notebooks/01_exploration.ipynb) + `src/audit.py`, `src/video_quality.py` | profiling, anomaly detection, and findings on both modalities |
| 2. Labeling design | [`docs/02_labeling_design.md`](docs/02_labeling_design.md) | 3-level schema, tooling, inter-annotator agreement, temporal alignment |
| 3. Curation pipeline | `src/curate.py` (+ `src/load.py`) | filter → clean → save curated subset + decision report |
| 4. Policy evaluation design | [`docs/04_eval_design.md`](docs/04_eval_design.md) | metrics, rollout counts, sim-to-real diagnosis, ego success detector |
| 5. VLA adaptation (bonus) | [`docs/05_vla_adaptation.md`](docs/05_vla_adaptation.md) | OpenVLA fine-tuning: format, hyperparameters, egocentric failure modes |

## Data

Two public LeRobot datasets, chosen so both modalities are actually exercised:

- **`lerobot/aloha_sim_insertion_human`** — the dataset named in the prompt.
  50 sim episodes, 14 joints, 50 fps, one top camera. It has **no wrist
  camera**, so for the egocentric half I added:
- **`lerobot/svla_so100_pickplace`** — 50 real SO100 episodes, 30 fps,
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
the real files. Fewer deps, and it forced me to actually understand the format.

**Audit checks are pure functions, used twice.** The notebook explores with
exactly the code the pipeline filters with — the analysis can't drift from
the production behavior.

**Episodes get dropped; frames get masked.** Deleting individual video frames
breaks the 1:1 temporal alignment with the joint stream, so frame-level
defects (blur, clipping) are recorded as masks for the training dataloader to
skip, and only episode-level defects cause deletions. Videos are never
re-encoded — curated data points back into the source mp4s via timestamps.

**Thresholds are physical/robust, not absolute.** The two datasets disagree
about everything absolute (radians vs degrees, float-jitter vs 12-bit
quantized encoders, 50 vs 30 fps). Jump detection uses MAD + a physical floor
("no joint sweeps its full range in <0.25 s"); blur thresholds are relative to
each stream's own median. Both of these were bugs first: v1 of the jump check
flagged 500 false positives per episode, and v1 of the stall check flagged
every svla episode because quantized encoders legitimately repeat values. The
notebook documents both iterations.

**Deterministic and reproducible.** Same input → byte-identical parquet
(verified by hash). The report embeds its config, so any curated set can
answer "what settings produced you?"

### What the audits actually found

- **aloha**: 8/50 episodes contain single-frame teleports (one joint moves an
  implausible amount in 20 ms) → dropped. Zero NaNs / timing gaps.
- **svla**: median episode is ~19% fully-stalled (operator pauses — 
  indistinguishable from bus dropout using positions alone); 4 episodes exceed
  30% dead time → dropped. Wrist-cam blur is ~10× worse than the top cam and
  bursts during fast reaches (up to 9% of frames) → masked, episodes kept.

## Teleop vs egocentric: the key trade-offs

| | joint states | egocentric video |
|---|---|---|
| defect definition | physically impossible | perceptually useless |
| thresholds | physical limits + robust stats | relative to the stream's own stats |
| filter granularity | episode drops | frame masks (episode drop only >20% bad) |
| aggressiveness | strict — bad state is *wrong* | conservative — blurry frame is still *informative* |
| the trap | unit/encoder assumptions | filtering blur selects for slow demos and biases the policy slow |

## What I'd do next (more time / hardware access)

1. **Task-space checks** — audit end-effector paths via forward kinematics,
   not just joint space; catches near-collision and workspace-exit defects
   that per-joint checks can't see.
2. **Occlusion detection on the wrist cam** — gripper-occupancy segmentation;
   blur and exposure are solved cheaply, occlusion is the remaining
   quality axis and it matters most exactly at grasp time.
3. **Close the loop** — the Task 4 egocentric success detector, trained on
   Task 2 labels, auto-scoring incoming episodes so curation happens at
   collection time instead of after.
4. **On real hardware**: bus-level comms logging (the only way to tell sensor
   dropout from operator pauses) and camera-to-joint clock calibration —
   this pipeline *trusts* LeRobot's timestamps; a collection rig has to *earn*
   them.

## Assumptions

- Success/task labels don't exist in these datasets, so curation is purely
  quality-based; with labels, outcome-aware filtering comes first.
- Curated output stays in parquet + source-video references rather than a
  re-packaged LeRobot dataset (faster to iterate; conversion is mechanical).
- Wrist cam ≈ egocentric. True for OpenArm's setup as described; a head-mounted
  rig would add gaze/parallax questions this design doesn't cover.
