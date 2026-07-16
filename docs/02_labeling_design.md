# Task 2: Data Labeling Design

Design for labeling pick-and-place teleoperation episodes, using
`svla_so100_pickplace` ("pick up the cube and place it in the box") as the
reference throughout.

## Teleoperation side

### Label schema: three levels, cheapest first

**Episode-level** (every episode gets these, and each one takes seconds to assign):

| label | type | why |
|---|---|---|
| `outcome` | success / failure / partial | the one label every downstream consumer needs |
| `failure_mode` | dropped / missed-grasp / wrong-place / knocked-over / other | only assigned when the outcome is not a success, and it feeds both eval and data collection priorities |
| `operator_quality` | clean / hesitant / corrected | "corrected" means the operator recovered mid-episode, and those recovery segments are valuable training data, so the label preserves them instead of treating them as defects |
| `retake` | bool | the operator flagged it themselves at collection time, which makes it about the cheapest label in existence |

**Segment-level** (the workhorse, spans of frames tagged with a phase):

```
approach → grasp → transport → place → retreat   (+ recovery, idle)
```

Phase boundaries are proposed automatically and then corrected by humans. The
gripper command and end-effector speed already segment a pick-and-place fairly
well on their own. The `grasp` phase starts when the gripper begins closing,
`transport` begins when speed rises while the gripper stays closed, and `idle`
comes straight from the stall detector already built in `src/audit.py`. Because
annotators are fixing proposed boundaries rather than drawing them from scratch,
the work goes five to ten times faster, and it converts labeling from a drawing
task into a reviewing one, which is also much easier to QA.

**Frame-level**: none by default. Frame labels cost 100 to 500 times what
episode labels cost, and imitation learning mostly does not need them. The two
cases where they would help, `contact` moments and `grasp_settled`, are both
derivable from the segment boundaries anyway.

### Tooling

I would use Label Studio for the episode-level tags, since its video player and
form UI are built for exactly this kind of job, and a custom review script for
the segment boundaries. That script would show matplotlib joint traces and
gripper state underneath the synced video, with keyboard nudging of the
auto-proposed boundaries. I would normally resist building a custom tool, but
the propose-then-correct loop is the whole efficiency win here, and no
off-the-shelf tool renders joint traces beside video. If we later need per-frame
object boxes or masks, CVAT is still the answer.

### Inter-annotator agreement for motion-based labels

The tricky part with motion labels is that phase boundaries are inherently
fuzzy. It is genuinely hard to say when "approach" becomes "grasp," so naive
frame-exact agreement will look terrible even between two careful annotators.

For the categorical labels like outcome and failure_mode, I would use Cohen's
kappa with a target of κ > 0.8. If agreement falls below that, the problem is
usually that the label definitions are ambiguous, so the fix is to tighten the
rubric rather than retrain the annotators.

For segment boundaries, I would use agreement-at-tolerance, where two boundaries
count as agreeing if they fall within ±0.25 s of each other, which matches the
timescale of the motions themselves. Rather than reporting a single number, I
would report the whole agreement curve at ±0.1, 0.25, and 0.5 s, so the
fuzziness becomes something we can see and reason about instead of argue over.

The protocol around all of this is to double-label 10% of episodes, adjudicate
disagreements weekly, and fold each resolved example back into the rubric as a
new gold standard.

## Egocentric side

The wrist camera needs scene-semantic labels that the joint stream simply
cannot express, things like what is visible, what is being touched, and what
went wrong.

| label | level | notes |
|---|---|---|
| `object_visible` / `target_visible` | segment | occlusion spans fall out as the complement |
| `interaction` | segment | none / touching / holding / releasing, describing hand-object interaction states rather than robot phases |
| `failure_moment` | timestamp | the frame where it went wrong, whether a slip or a miss, which is often visible only in the wrist view |
| `gaze_proxy` | segment, coarse | what the camera is "looking at": workspace, object, target, or clutter. A wrist cam is essentially a pointing device, so where the operator aims it correlates with attention |
| `visual_quality` | frame, automatic | blur, exposure, and freeze masks from `src/video_quality.py`. These are never hand-labeled, since the pipeline already computes them |

The `interaction` label deliberately overlaps with the teleop `grasp` and
`transport` phases. The joint stream tells us what the arm did, and the video
tells us what the object did. When a grasp phase shows up with
`interaction: none`, that is a missed grasp, and the disagreement between the
two label tracks is itself the signal. It is exactly the eval signal that
Task 4 relies on.

## Temporal alignment between the two label tracks

There is one rule that makes this whole thing work: every label attaches to a
timestamp rather than a frame index. LeRobot already gives every joint row and
every video frame a shared per-episode clock (the `timestamp` column lines up
with the video PTS), so a few things follow for free.

A label spanning `[t0, t1)` projects onto joint rows wherever
`t0 ≤ timestamp < t1` and onto video as `round(t * fps)`, with no per-modality
bookkeeping required. Streams running at different rates (joints at 50 Hz, video
at 30 fps, a future depth cam at 15 fps) all inherit the same labels
automatically. And because labels live on timestamps, curation that drops frames
can never orphan a label. The Task 3 pipeline refuses to drop individual frames
anyway, for this exact reason.

One sequencing decision is worth calling out: labeling happens after curation,
on the curated set, so no annotation budget gets spent on episodes the filter
was going to delete. The one exception is episodes dropped for being `stalled`.
Those get a cheap episode-level look, because the question of why the operator
froze is really feedback about the collection process, and that feedback is
often worth more than the labels themselves.
