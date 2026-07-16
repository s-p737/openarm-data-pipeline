# Task 4: Policy Evaluation Design

Setting: an ACT or Diffusion Policy trained on the curated pick-and-place set
from Task 3. How do we know if it is any good?

## Teleoperation side

### Protocol

Here are the metrics, listed in order of how much I actually trust them.

1. **Success rate.** A binary call, judged against a written rubric that is
   decided before evaluation starts ("cube fully inside box at t_end + 2 s,
   nothing else displaced"). Vague success criteria are the single biggest
   reason eval numbers inflate, so the rubric has to be pinned down first.
2. **Time-to-completion** (successes only). This is what separates a policy that
   barely works from one that works cleanly, and regressions tend to show up
   here first, while the success rate still looks flat.
3. **Progress score.** How far along the phase sequence the attempt actually got
   (reached, grasped, transported, placed). Knowing where a failure happens is
   worth far more than knowing how many there were. "80% of failures die at
   grasp" tells you exactly what to fix, whereas "60% success" tells you almost
   nothing.
4. **Intervention count**, if a human is babysitting long rollouts.

On sample size, I would run at least 50 rollouts per condition. At n=50, a
70%-success policy carries a ±13% confidence interval at 95%, and that is really
the floor for claiming two checkpoints differ at all. The 10-rollout evals that
show up in a lot of papers have intervals closer to ±30%, which cannot tell 60%
apart from 85%. When I need to rank several checkpoints cheaply, I would run 20
rollouts each as a first screen and then 50 or more on the top two.

Conditions have to be controlled rather than eyeballed. That means a fixed grid
of object start poses (say 5 positions by 2 orientations), fixed lighting, and
results reported per condition bucket instead of only in aggregate, because the
aggregate quietly hides something like "always fails in the far-left cell."
Everything gets logged the way a training run would be: checkpoint hash, seed,
initial pose, and per-rollout observations saved so any rollout can be replayed.

### Diagnosing "works in sim, fails on the real robot"

I would work from the cheapest experiment upward, since each step isolates one
seam of the sim-to-real gap.

1. **Replay real observations offline** through the policy, with no robot in the
   loop. If the actions look insane on real images but fine on sim images, the
   gap is visual. If they look fine on both, the problem lives downstream in
   dynamics or control.
2. **Diff the observation distributions.** Compare image stats (brightness,
   contrast, white balance) and joint-state distributions between sim and real.
   This is also where the classic silent killers hide: normalization constants,
   channel order (BGR versus RGB), resize interpolation, and proprioception
   units. Degrees versus radians took down half the audits in Task 1, and it
   will happily take down a policy too.
3. **Check the control seam.** Look at command-to-execution latency, controller
   gains, and action clipping. Sim executes instantly, so a real arm carrying
   80 ms of latency can turn a marginal policy into an oscillating one.
4. **Run an open-loop sanity test.** Replay a recorded demo's actions on the
   real robot with no policy involved. If even that fails, the gap is
   calibration or hardware, and no amount of policy work will fix it.
5. Only after all of that would I reach for domain randomization or co-training
   with real data.

## Egocentric side

### What the wrist camera changes

Joint streams can look completely successful even if the task fails. The
arm can trace the right path, close the gripper on air, and "place" nothing at
all. Joint states measure the robot, but the task actually happens to the
object, and the wrist camera is the one sensor pointed at the object.

The failure modes that are visible only in egocentric video:

- **Grasped air.** The gripper closed at the right pose, but the object was
  never in it.
- **In-hand slip or rotation.** The object drifts during transport, and the
  joints notice nothing until the placement misses.
- **Wrong object** (in clutter). A correct trajectory carried out on the wrong
  target.
- **Near-misses.** The attempt succeeded, but the object teetered on the box
  edge. Joint metrics score this as a clean success while the video shows
  something about to start failing, which makes it the earliest regression
  signal you can get.
- **Placement quality.** Landing in the box versus bouncing off the rim versus
  wedging at an angle.

So the eval protocol picks up three additions: record the wrist stream on every
rollout, review the video for every failure (with the Task 2 `failure_moment`
labels pointing to where it went wrong), and spot-check a sample of the
successes for near-miss signatures.

### Bonus: egocentric success detector

This would be a small binary classifier running on the last two seconds or so of
wrist frames.

For the backbone, I would use frozen pretrained ViT or ResNet embeddings with a
light head on top. At around 50 labeled episodes this is a few-hundred-example
problem, so there is no case for fine-tuning the trunk. A zero-shot VLM scoring
"cube in box" CLIP-style makes a good baseline to beat.

The labels come from the Task 2 episode `outcome` labels, since the labeling
schema was designed so that this classifier gets its training data for free.

I would pool embeddings across the window rather than scoring a single frame,
because a single final frame tends to get occluded by the gripper at exactly the
wrong moment. Validation runs against human judgment on held-out rollouts,
reported as precision and recall. A success detector that overcounts successes
is worse than having none at all, because it silently corrupts every number
built on top of it.

The payoff compounds over time. First the detector auto-scores overnight eval
runs, then the same model filters and labels newly collected demos as a form of
auto-curation, and eventually it can serve as a reward signal if we ever go down
the RL-fine-tuning route.
