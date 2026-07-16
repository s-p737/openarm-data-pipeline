# Task 4 — Policy Evaluation Design

Setting: an ACT or Diffusion Policy trained on the curated pick-and-place set
from Task 3. How do we know if it's any good?

## Teleoperation side

### Protocol

**Metrics, in order of how much I trust them:**

1. **Success rate** — binary, judged against a written rubric decided *before*
   evaluation ("cube fully inside box at t_end + 2 s, nothing else displaced").
   Vague success criteria are how eval numbers inflate.
2. **Time-to-completion** (successes only) — separates "barely works" from
   "works"; regressions show up here first while success rate still looks flat.
3. **Progress score** — how far along the phase sequence the attempt got
   (reached / grasped / transported / placed). Failure *location* is worth more
   than failure *count*: "80% of failures die at grasp" tells you what to fix,
   "60% success" doesn't.
4. **Intervention count** if a human babysits long rollouts.

**How many rollouts:** 50 per condition minimum. With n=50, a 70%-success
policy has a ±13% (95%) confidence interval — that's the *floor* for claiming
two checkpoints differ. 10-rollout evals (common in papers) have ±30-ish
intervals and can't distinguish 60% from 85%. When ranking multiple
checkpoints cheaply: 20 rollouts each as a screen, then 50+ on the top two.

**Conditions must be controlled, not vibes:** a fixed grid of object start
poses (e.g. 5 positions x 2 orientations), fixed lighting, results reported
**per condition bucket**, not just aggregate — aggregate hides "always fails
in the far-left cell". Everything logged like a training run: checkpoint hash,
seed, initial pose, per-rollout observations saved for replay.

### Diagnosing "works in sim, fails on the real robot"

Cheapest experiment first — each step isolates one seam of the sim-to-real gap:

1. **Replay real observations offline** through the policy (no robot). If
   actions look insane on real images but fine on sim images → *visual* domain
   gap. If fine on both → the problem is downstream: dynamics or control.
2. **Diff the observation distributions** — image stats (brightness, contrast,
   white balance) and joint-state distributions, sim vs real. Also the classic
   silent killers: normalization constants, channel order (BGR/RGB!), resize
   interpolation, proprioception units — degrees vs radians took down half the
   audits in Task 1, it will happily take down a policy too.
3. **Check the control seam** — command→execution latency, controller gains,
   action clipping. Sim executes instantly; a real arm with 80 ms of latency
   turns a marginal policy into an oscillating one.
4. **Open-loop sanity test** — replay a *recorded demo's actions* on the real
   robot with no policy. If that fails, the gap is calibration/hardware, and no
   amount of policy work fixes it.
5. Only after 1–4: domain randomization / co-training with real data.

## Egocentric side

### What the wrist camera changes

Joint streams can look completely successful while the task fails — the arm
traced the right path, closed the gripper on air, and "placed" nothing. Joint
states measure the *robot*; the task happens to the *object*, and the wrist
camera is the sensor pointed at the object.

Failure modes visible only in egocentric video:

- **grasped air** — gripper closed at the right pose, object not in it
- **in-hand slip / rotation** — object drifts during transport; joints notice
  nothing until placement misses
- **wrong object** (clutter) — correct trajectory, wrong target
- **near-misses** — succeeded, but the object teetered on the box edge; joint
  metrics call this a clean success, video calls it "about to start failing",
  and it's the earliest regression signal you can get
- **placement quality** — in the box vs bounced off the rim vs wedged at an angle

So the eval protocol adds: record the wrist stream on every rollout, review
video for every failure (with the Task 2 `failure_moment` labels giving the
where), and spot-check a sample of *successes* for near-miss signatures.

### Bonus: egocentric success detector

A small binary classifier on the last ~2 s of wrist frames:

- **Backbone**: frozen pretrained ViT/ResNet embeddings, light head on top —
  at ~50 labeled episodes this is a few-hundred-example problem, so no
  fine-tuning of the trunk. Zero-shot VLM (CLIP-style "cube in box" scoring)
  as the baseline to beat.
- **Labels**: the Task 2 episode `outcome` labels — the labeling schema was
  designed so this comes free.
- **Multi-frame, not single-frame**: mean-pool embeddings over the window;
  single final frames get occluded by the gripper at exactly the wrong moment.
- **Validation**: against *human* judgment on held-out rollouts, reported as
  precision/recall (a success detector that overcounts successes is worse than
  useless — it corrupts every number built on top of it).

Payoff compounds: auto-scored overnight eval runs first, then the same
detector filters/labels newly collected demos (auto-curation), and eventually
becomes a reward signal if we ever go the RL-fine-tuning route.
