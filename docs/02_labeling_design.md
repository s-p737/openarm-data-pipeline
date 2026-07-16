# Task 2 — Data Labeling Design

Design for labeling pick-and-place teleoperation episodes, using
`svla_so100_pickplace` ("pick up the cube and place it in the box") as the
concrete reference throughout.

## Teleoperation side

### Label schema — three levels, cheapest first

**Episode-level** (every episode gets these — seconds each to assign):

| label | type | why |
|---|---|---|
| `outcome` | success / failure / partial | the one label every downstream consumer needs |
| `failure_mode` | dropped / missed-grasp / wrong-place / knocked-over / other | only when outcome ≠ success; feeds eval + data collection priorities |
| `operator_quality` | clean / hesitant / corrected | "corrected" = operator recovered mid-episode; recovery segments are *valuable* training data, not defects |
| `retake` | bool | operator flagged it themselves at collection time — cheapest label in existence |

**Segment-level** (the workhorse — spans of frames tagged with a phase):

```
approach → grasp → transport → place → retreat   (+ recovery, idle)
```

Phase boundaries are **proposed automatically, corrected by humans**. The
gripper command + end-effector speed already segment a pick-and-place pretty
well: `grasp` starts when the gripper begins closing, `transport` when speed
rises with the gripper closed, `idle` comes straight from the stall detector
already built in `src/audit.py`. Annotators fix boundaries instead of drawing
them from scratch — 5–10x faster, and it converts labeling from "drawing" to
"reviewing", which is also much easier to QA.

**Frame-level**: none by default. Frame labels are 100–500x the cost of
episode labels, and imitation learning mostly doesn't need them — the two
exceptions (`contact` moments, `grasp_settled`) are derivable from
segment boundaries.

### Tooling

**Label Studio** for episode-level tags (its video player + form UI is exactly
this job) and a **custom review script** for segment boundaries: matplotlib
joint traces + gripper state under the synced video with keyboard nudging of
auto-proposed boundaries. I'd normally resist "custom tool", but the
auto-propose-then-correct loop is the whole efficiency win and no off-the-shelf
tool renders joint traces beside video. CVAT stays the answer if we later need
per-frame object boxes/masks.

### Inter-annotator agreement for motion-based labels

The trap with motion labels: phase boundaries are inherently fuzzy (when
exactly does "approach" become "grasp"?), so naive frame-exact agreement will
look terrible even between two careful annotators.

- **Categorical labels** (outcome, failure_mode): Cohen's kappa, target κ > 0.8.
  Below that, the label *definitions* are ambiguous — fix the rubric, not the
  annotators.
- **Segment boundaries**: agreement@tolerance — two boundaries "agree" if
  within ±0.25 s (matches the timescale of the motions themselves). Report the
  agreement curve at ±0.1 / 0.25 / 0.5 s rather than one number, so we can *see*
  fuzziness instead of arguing about it.
- **Protocol**: 10% of episodes double-labeled, disagreements adjudicated
  weekly, rubric updated with the resolved examples as goldens.

## Egocentric side

The wrist camera needs *scene-semantic* labels the joint stream can't express
— what is visible, what is being touched, what went wrong:

| label | level | notes |
|---|---|---|
| `object_visible` / `target_visible` | segment | occlusion spans fall out as the complement |
| `interaction` | segment | none / touching / holding / releasing — hand-object interaction states, not robot phases |
| `failure_moment` | timestamp | the frame where it went wrong (slip, miss) — often visible ONLY in the wrist view |
| `gaze_proxy` | segment, coarse | what the camera is "looking at": workspace / object / target / clutter. A wrist cam is a pointing device — where the operator aims it correlates with attention |
| `visual_quality` | frame, automatic | blur/exposure/freeze masks from `src/video_quality.py` — never hand-labeled, the pipeline already computes them |

`interaction` deliberately overlaps with the teleop `grasp/transport` phases:
the joint stream says what the *arm* did, the video says what the *object*
did. A grasp phase with `interaction: none` is a missed grasp — the
disagreement between the two label tracks is itself signal, and it's exactly
the eval signal Task 4 uses.

## Temporal alignment between the two label tracks

One rule: **all labels attach to timestamps, not frame indices.** LeRobot
already gives every joint row and video frame a shared per-episode clock
(`timestamp` column ↔ video PTS), so:

- a label `[t0, t1)` projects onto joint rows as `t0 ≤ timestamp < t1` and
  onto video as `round(t * fps)` — no per-modality bookkeeping
- streams at different rates (joints at 50 Hz, video at 30 fps, a future depth
  cam at 15 fps) inherit the same labels for free
- curation that drops frames can't orphan labels (and the Task 3 pipeline
  refuses to drop individual frames anyway, for exactly this reason)

Labeling happens **after** curation, on the curated set — no budget spent
annotating episodes the filter was going to delete. The one exception:
episodes dropped for `stalled` get a cheap episode-level look, because "why
did the operator freeze here?" is collection-process feedback, and that
feedback is worth more than the labels themselves.
