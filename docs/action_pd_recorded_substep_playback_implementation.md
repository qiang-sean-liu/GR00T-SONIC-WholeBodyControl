# Action-PD Recorded Substep Playback Implementation

This note documents the implementation of exact recorded-substep playback in
`gear_sonic/scripts/playback_lerobot_action_pd.py`.

The relevant commands are:

```bash
./.venv_sim/bin/python gear_sonic/scripts/playback_lerobot_action_pd.py \
  --dataset_dir outputs/2026-05-04-02-10-47-lowcmd-debug \
  --episode 2 \
  --env_name default \
  --fps 50 \
  --free_base \
  --use_recorded_substep_inputs \
  --init_substep_frame 1 \
  --init_substep_index 0 \
  --init_substep_all_inputs \
  --start_frame 1
```

and:

```bash
./.venv_sim/bin/python gear_sonic/scripts/playback_lerobot_action_pd.py \
  --dataset_dir outputs/2026-05-04-01-45-59-lowcmd-debug \
  --episode 0 \
  --env_name default \
  --fps 50 \
  --free_base \
  --use_recorded_substep_inputs \
  --init_substep_frame 1 \
  --init_substep_index 0 \
  --init_substep_all_inputs \
  --start_frame 1
```

## Goal

The goal was to replay the MuJoCo recording as exactly as possible, not only at
the LeRobot frame level, but at every MuJoCo substep inside each frame.

Frame-level data such as `observation.state` and `action.wbc` is not enough for
exact replay. MuJoCo advances with internal state and inputs such as `qpos`,
`qvel`, `ctrl`, applied forces, and warm-start acceleration. If any of those
values differ, the next physics state can diverge, especially during contact
with the cube.

## Recorded Substep Data

The lowcmd debug recording stores the data needed to reproduce the MuJoCo
substep stream:

- `robot.mujoco_substep_qpos`: recorded MuJoCo `qpos` before each substep.
- `robot.mujoco_substep_qvel`: recorded MuJoCo `qvel` before each substep.
- `robot.mujoco_substep_ctrl`: recorded actuator `ctrl`.
- `robot.mujoco_substep_qfrc_applied`: recorded generalized applied force.
- `robot.mujoco_substep_xfrc_applied`: recorded body external force.
- `robot.mujoco_substep_qacc_warmstart`: recorded warm-start acceleration.
- `robot.motor_pd_substep_sim_time`: recorded MuJoCo simulation time for each substep.

These fields are recorded per LeRobot frame, but each row contains multiple
MuJoCo substeps.

## Initialization From f1s0

The command initializes playback from recorded frame `1`, substep `0`:

```bash
--init_substep_frame 1
--init_substep_index 0
--start_frame 1
```

Inside `playback_lerobot_action_pd.py`, this loads:

- `robot.mujoco_substep_qpos[f1][s0]` into `data.qpos`;
- `robot.mujoco_substep_qvel[f1][s0]` into `data.qvel`.

With:

```bash
--init_substep_all_inputs
```

it also loads the recorded initial inputs:

- `data.time`;
- `data.ctrl`;
- `data.qfrc_applied`;
- `data.xfrc_applied`;
- `data.qacc_warmstart`.

This means playback starts from the same MuJoCo state and input context that was
recorded at `f1s0`.

## Recorded Substep Playback Mode

The key flag is:

```bash
--use_recorded_substep_inputs
```

In this mode, playback does not recompute the substep inputs from frame-level
actions. Instead, for each recorded substep it restores exactly what was saved:

```text
data.time
data.qpos
data.qvel
data.ctrl
data.qfrc_applied
data.xfrc_applied
data.qacc_warmstart
```

Then it calls:

```text
mujoco.mj_step(model, data)
```

So the environment is still a real MuJoCo physics environment. The robot and
cube interact through collision/contact physics, but the state and inputs going
into each MuJoCo substep are the exact recorded values.

## Overlap Handling

Recorded frame windows can overlap because each LeRobot frame can carry a small
window of substeps. The playback code treats substeps as one monotonic stream.

If a recorded substep time is older than the current `data.time`, it is skipped.
This prevents replay from stepping backward or repeating stale substeps.

## Free Base

The command uses:

```bash
--free_base
```

This disables driving the base from frame-level recorded base pose during normal
Action-PD simulation. In recorded-substep-input mode, the base state is already
contained in recorded `qpos/qvel`, so the substep replay follows the recorded
MuJoCo free-base state directly.

## Why This Matters

This implementation lets us check whether playback can match the recording at
the MuJoCo substep level. It avoids ambiguity from asynchronous frame-level
signals such as `observation.state`, `action.wbc`, or `g1_debug`.

For cube interaction, this is important because contact is sensitive to the
exact robot state and applied forces. By restoring the recorded substep state
and inputs, playback gives MuJoCo the same conditions that existed during
recording, so robot/cube interaction is reproduced as closely as the recorded
data allows.

## Summary

`playback_lerobot_action_pd.py --use_recorded_substep_inputs` is the exact
recorded-substep playback path.

It initializes from recorded `f1s0`, restores recorded MuJoCo state and inputs,
skips overlapping stale substeps, and advances MuJoCo with `mj_step` so the
robot still physically interacts with the cube in the environment.
