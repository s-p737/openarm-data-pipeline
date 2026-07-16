# Task 5 (Bonus): Fine-tuning a VLA on OpenArm Data

Option A: a VLA, using OpenVLA as the concrete reference (open weights, open
fine-tuning recipe, and the same logic carries over to pi0-style models).

## Teleoperation side

### Data format

OpenVLA consumes `(image, language instruction, action)` triples, and RLDS is
its native format. The LeRobot to RLDS conversion is mechanical, since our
pipeline already produces exactly the pieces it needs per frame: the camera
frame, the task string (from `meta/tasks.parquet`, "Pick up the cube and place
it in the box"), and the action vector. Two format decisions matter more than
the container itself.

The first is the action space. OpenVLA predicts actions as discretized tokens,
where each action dimension is binned into 256 buckets over the dataset's 1st to
99th percentile range. Those normalization statistics become part of the model,
so they have to be computed on the curated set from Task 3 instead of the raw
data. The 8 teleport episodes we dropped would have stretched the bins and
wasted resolution on motion that never actually happened. This is the point where
data curation turns directly into model quality.

The second is control frequency. The pretrained model mostly saw data around 3
to 10 Hz, while our streams run at 30 to 50 Hz. I would subsample down to
roughly 5 to 10 Hz for fine-tuning rather than feeding raw 50 Hz, because
consecutive frames at that rate are near-duplicates and the per-step action
deltas shrink toward bin-resolution noise.

### Key hyperparameters (in order of how likely they are to cause issues)

1. **Action normalization bounds.** As above. Wrong bounds mean the action
   range gets silently clipped or wasted.
2. **LoRA versus full fine-tune.** LoRA at around r=32 is the standard approach
   for small datasets. Touching all the weights with only 46 episodes mostly just
   erases what the model already knew.
4. **Learning rate.** Keep it small (roughly 5e-4 for LoRA, 2e-5 for full) on a
   short cosine schedule. Overshooting erases the pretrained visuomotor priors,
   which were the whole reason to reach for a VLA in the first place.
5. **Epochs.** With a tiny dataset and a big model, I would lean on heavy
   augmentation and early-stop on rollout success (the Task 4 protocol) rather
   than on action-token accuracy, which plateaus early and stops tracking actual
   task success.
6. **Image resolution.** Match the pretraining size (224×224) exactly. A resize mismatch fails silently and drags down performance.

## Egocentric side

### Preprocessing and aligning the wrist stream

Temporal alignment is already solved upstream. LeRobot's shared timestamp clock
pairs each wrist frame with its action row, and Task 3 preserved that pairing
1:1, so subsampling to 5 to 10 Hz picks the (frame, action) pairs together.

I would skip the blurred frames at training time rather than worry about them
earlier. Bad frames were masked instead of deleted back in Task 3, so the
dataloader can skip just those pairs without breaking the sequence structure. At
5 Hz subsampling we are throwing out 80% of frames anyway, so it costs nothing to
preferentially keep the sharp ones.

Everything gets the same resize, crop, and normalization as the pretrained
vision encoder, plus photometric augmentation like brightness and contrast
jitter, because wrist-cam exposure swings are a train-test mismatch waiting to
happen.

### Failure modes: a third-person pretrained model meets egocentric input

This is a distribution shift instead of just a different camera angle.

1. **Viewpoint prior mismatch.** The pretraining data is overwhelmingly static
   third-person scenes where the robot arm is visible in frame. In a wrist view
   the arm is invisible, since you are effectively standing on it, and the whole
   scene moves with every joint. The model's learned visual grounding ("arm is
   left of the object, so move right") is not just missing here, it is actively
   wrong.
2. **Self-occlusion at the decisive moment.** The gripper fills the frame right
   at grasp time. A third-person model never had its view blocked by its own
   body, so it can misread the occlusion as a change in the scene.
3. **Apparent motion confusion.** Camera ego-motion looks like world motion.
   Features that used to mean "object is moving" now fire constantly while the
   object sits still.
4. **Absent global context.** The wrist view often cannot see the target while
   it is looking at the object, and a third-person-trained model has no habit of
   remembering an off-screen goal.

I would try the mitigations in this order. First, dual-input fine-tuning: keep
the top camera as the primary view and add the wrist stream as a second image
token stream, so the egocentric input enriches the priors instead of replacing
them, which is what pi0-style multi-cam setups do. Second, longer fine-tuning
aimed at the vision encoder specifically, meaning LoRA on the vision layers and
not only the LLM blocks, since the shift is visual rather than semantic. Third,
if wrist-only is a hard requirement, warm up on an egocentric-rich mix such as
DROID wrist streams before the task fine-tune.

An honest note on data volume: 46 curated episodes of a single task is enough
for LoRA to specialize a VLA on that task, and nobody should expect
generalization claims out of it. The realistic near-term win is the pipeline
itself. Curated, aligned, mask-aware data means every future episode we collect
is immediately trainable. The model story gets better with scale, but the data
story had to be right from the start.
