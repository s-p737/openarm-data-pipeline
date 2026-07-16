# Task 5 (Bonus) — Fine-tuning a VLA on OpenArm Data

Option A: VLA, using OpenVLA as the concrete reference (open weights, open
fine-tuning recipe; the same logic applies to pi0-style models).

## Teleoperation side

### Data format

OpenVLA consumes `(image, language instruction, action)` triples; RLDS is its
native format, and LeRobot→RLDS conversion is mechanical since our pipeline
already produces exactly the needed pieces per frame: camera frame + task
string (`meta/tasks.parquet` — "Pick up the cube and place it in the box") +
action vector. Two format decisions matter more than the container:

- **Action space**: OpenVLA predicts actions as discretized tokens — each
  action dimension binned into 256 buckets over the *dataset's* 1st–99th
  percentile range. Those normalization statistics become part of the model.
  Compute them on the **curated** set (Task 3), not raw — the 8 teleport
  episodes we dropped would have stretched the bins and wasted resolution on
  motion that never really happened. This is where data curation literally
  becomes model quality.
- **Control frequency**: the pretrained model saw mostly ~3–10 Hz data; our
  streams are 30–50 Hz. Subsample to ~5–10 Hz for fine-tuning rather than
  feeding raw 50 Hz (near-duplicate consecutive frames, and per-step action
  deltas shrink toward bin-resolution noise).

### Key hyperparameters (in order of how likely they are to bite)

1. **Action normalization bounds** — see above; wrong bounds = silently
   clipped or wasted action range.
2. **LoRA vs full fine-tune** — LoRA (r=32-ish) is the published recipe for
   small datasets; with only 46 curated episodes full fine-tuning mostly buys
   catastrophic forgetting.
3. **Learning rate** — small (~5e-4 LoRA / 2e-5 full), short cosine schedule;
   overshooting erases pretrained visuomotor priors, which were the reason to
   use a VLA at all.
4. **Epochs** — tiny dataset, big model: heavy augmentation, early stopping on
   *rollout* success (Task 4 protocol), not on action-token accuracy — which
   plateaus early and stops correlating with actual task success.
5. **Image resolution** — match pretraining (224×224) exactly; resize
   mismatches are silent performance killers.

## Egocentric side

### Preprocessing / aligning the wrist stream

- Temporal alignment is already solved upstream: LeRobot's shared timestamp
  clock pairs each wrist frame with its action row (Task 3 preserved this
  1:1). Subsampling to 5–10 Hz picks aligned (frame, action) pairs together.
- **Skip the blurred-frame mask at training time** — bad frames were masked,
  not deleted (Task 3), so the dataloader can skip *just* those pairs without
  breaking sequence structure. At 5 Hz subsampling we're discarding 80% of
  frames anyway; preferentially keep sharp ones.
- Same resize/crop/normalization as the pretrained vision encoder; add
  photometric augmentation (brightness/contrast jitter) because wrist-cam
  exposure swings are a *train-test* mismatch waiting to happen.

### Failure modes: third-person pretrained model meets egocentric input

This is a genuine distribution shift, not just "different camera":

1. **Viewpoint prior mismatch** — pretraining data is overwhelmingly static
   third-person scenes where the robot arm is *visible in frame*. In a wrist
   view the arm is invisible (you're standing on it) and the whole scene moves
   with every joint. The model's learned visual grounding ("arm is left of
   object → move right") is not just missing, it's *wrong*.
2. **Self-occlusion at the decisive moment** — the gripper fills the frame
   exactly at grasp time. A third-person model never had its view blocked by
   its own body; it can misread occlusion as scene change.
3. **Apparent motion confusion** — camera ego-motion looks like world motion.
   Features that meant "object moving" now fire constantly while the object is
   still.
4. **Absent global context** — the wrist view often can't see the *target*
   while looking at the object; a third-person-trained model has no habit of
   "remembering" off-screen goals.

**Mitigations, in the order I'd try them:** (1) dual-input fine-tuning — keep
the top camera as primary and add the wrist stream as a second image token
stream, so egocentric enriches rather than replaces the priors (this is what
pi0-style multi-cam setups do); (2) longer fine-tuning on the vision encoder
specifically (LoRA on vision layers too, not just the LLM blocks) since the
shift is visual, not semantic; (3) if wrist-only is a hard requirement, warm up
with an egocentric-rich mix (e.g. DROID wrist streams) before task fine-tuning.

**Honest position on data volume:** 46 curated episodes of one task is enough
for LoRA to *specialize* a VLA on that task; nobody should expect
generalization claims from it. The realistic near-term win is the pipeline:
curated, aligned, mask-aware data means every future collected episode is
immediately trainable — the model story improves with scale, the data story
had to be right from episode one.
