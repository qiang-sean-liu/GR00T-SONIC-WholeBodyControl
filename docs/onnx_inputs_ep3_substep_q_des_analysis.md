# ONNX Inputs Episode 3 Substep `q_des` Analysis

Dataset:

```bash
outputs/onnx-inputs-debug/data/chunk-000/episode_000003.parquet
```

Playback command used for the investigation:

```bash
./.venv_sim/bin/python gear_sonic/scripts/playback_lerobot_onnx_inputs_pd_decoder_warmup.py \
  --dataset_dir outputs/onnx-inputs-debug \
  --episode 3 \
  --env_name default \
  --sonic_encoder gear_sonic_deploy/policy/release/model_encoder.onnx \
  --sonic_decoder gear_sonic_deploy/policy/release/model_decoder.onnx \
  --fps 50 \
  --free_base \
  --init_substep_frame 1 \
  --init_substep_index 0 \
  --init_substep_all_inputs \
  --start_frame 1 \
  --compare
```

## What Is Recorded

The MuJoCo simulator records one snapshot immediately before every 200 Hz `mj_step` call:

```python
pre_step_snapshot = self._make_body_pd_substep_snapshot(body_torques)
mujoco.mj_step(self.mj_model, self.mj_data)
self._record_body_pd_substep(pre_step_snapshot)
```

The pre-step snapshot includes:

- `robot.motor_pd_substep_q`
- `robot.motor_pd_substep_dq`
- `robot.motor_pd_substep_q_des`
- `robot.motor_pd_substep_dq_des`
- `robot.motor_pd_substep_kp`
- `robot.motor_pd_substep_kd`
- `robot.motor_pd_substep_tau_ff`
- `robot.motor_pd_substep_torque_raw`
- `robot.motor_pd_substep_torque`
- `robot.motor_pd_substep_sim_time`
- `robot.mujoco_substep_qpos`
- `robot.mujoco_substep_qvel`
- `robot.mujoco_substep_ctrl`
- `robot.mujoco_substep_qfrc_applied`
- `robot.mujoco_substep_xfrc_applied`
- `robot.mujoco_substep_qacc_warmstart`

Post-step results are also attached after `mj_step`:

- `robot.mujoco_substep_post_qpos`
- `robot.mujoco_substep_post_qvel`
- `robot.mujoco_substep_post_qacc`
- `robot.mujoco_substep_post_actuator_force`
- `robot.mujoco_substep_post_sim_time`

`run_data_exporter_onnx_inputs.py` does not recompute the substep data. It stores the arrays received from the simulator debug/base-state stream.

## Why 4 Substeps Are Present

There is no explicit loop that holds one action for four MuJoCo substeps.

The simulator runs at:

```text
SIMULATE_DT = 0.005 s = 200 Hz
```

The dataset/control frame is:

```text
0.02 s = 50 Hz
```

Therefore:

```text
0.02 / 0.005 = 4 substeps
```

The simulator keeps the latest 4 recorded 200 Hz snapshots:

```python
max_substeps = int(round(0.02 / self.sim_dt)) if self.sim_dt > 0 else 4
self._body_pd_substep_history = self._body_pd_substep_history[-max_substeps:]
```

This means the 4 substeps are an exporter grouping window, not an atomic command window.

## Why `q_des` Can Change Inside One Frame

The MuJoCo simulator reads the latest stored `LowCmd` every 200 Hz `sim_step`.

`compute_body_torques()` reads:

```python
self.unitree_bridge.low_cmd.motor_cmd[i].q
```

That `low_cmd` is updated asynchronously by DDS:

```python
def LowCmdHandler(self, msg):
    with self.low_cmd_lock:
        self.low_cmd = msg
        self.low_cmd_received = True
        self.new_low_cmd = True
```

On the C++ side, command publication runs in a separate 500 Hz thread:

```text
publish_dt_ = 0.002 s
control_dt_ = 0.02 s
```

So a new `LowCmd` can arrive between two 5 ms MuJoCo substeps. That is why a single 50 Hz recorded frame can contain different `q_des` values across its four 200 Hz substeps.

## Whole-Recording `q_des` Statistics

For episode 3:

```text
Total frames: 1125
Frames with exactly 4 valid q_des substeps: 1124
Frames with all 4 q_des substeps identical: 200
Frames with q_des changing inside the frame: 924
```

Percentages:

```text
constant q_des frames: 17.8%
changing q_des frames: 82.1%
```

First changed substep counts:

```text
first change at substep 1: 268 frames
first change at substep 2: 370 frames
first change at substep 3: 286 frames
```

Most common changed-step patterns:

```text
changed at substeps (2, 3):    370 frames
changed at substep  (3):       286 frames
changed at substeps (1, 2, 3): 268 frames
```

Examples of frames where all 4 `q_des` values are identical:

```text
3, 4, 11, 21, 30, 36, 39, 44, 46, 56, 60, 69
```

Frame 12 is not constant.

## Worst `q_des` Changes

Largest examples in this recording:

```text
Frame 486: max_abs=1.2759, l2=1.5416
Frame 432: max_abs=0.9686, l2=1.1520
Frame 29:  max_abs=0.7985, l2=0.9364
Frame 899: max_abs=0.5897, l2=0.7950
Frame 489: max_abs=0.5290, l2=0.7405
```

For these worst frames, recomputing body PD control with the recorded per-substep `q_des` reconstructs recorded body `ctrl` to about `1e-6`. This confirms that the per-substep `q_des` change is the parameter causing the recorded substep `ctrl` changes.

## Frame 12 Detailed Analysis

Frame 12 is zero-based frame index `11`.

Recorded `q_des` changes:

```text
s0: q_des_l2_vs_s0=0.000000e+00, max=0.000000e+00
s1: q_des_l2_vs_s0=5.720666e-02, max=3.020886e-02
s2: q_des_l2_vs_s0=5.720666e-02, max=3.020886e-02
s3: q_des_l2_vs_s0=5.720666e-02, max=3.020886e-02
```

### Substep 0

Playback starts from recorded `f12s0` exactly.

Pre-step inputs:

```text
sim_time diff:        0
qpos full diff:       0
qvel full diff:       0
body ctrl diff:       4.33e-07
left hand ctrl diff:  0
right hand ctrl diff: 0
qacc_warmstart diff:  0
body q_des diff:      0
```

Post-step result:

```text
post_qpos full diff: 1.42e-10
post_qvel full diff: 2.86e-08
post_qacc full diff: 5.84e-06
```

So substep 0 matches recording.

### Substep 1

Before step 1, playback still matches recorded `f12s1` state, but uses stale `q_des` from `s0`.

Key differences:

```text
body q_des diff:      5.720666e-02
body ctrl diff:       1.497257e+00
qacc_warmstart diff:  5.835495e-06
```

Post-step result:

```text
post_qpos full diff: 9.56e-04
post_qvel full diff: 2.01e-01
post_qacc full diff: 4.08e+01
```

The important point is that `qacc_warmstart` is still only `~5.8e-06` different before substep 1, while body `q_des` differs by `5.72e-02` and body `ctrl` differs by `1.497`. The divergence is caused by stale `q_des`, not by `qacc_warmstart`.

Largest substep-1 body control differences:

```text
right_knee_joint:          diff= 0.746011
waist_yaw_joint:           diff= 0.656801
right_hip_yaw_joint:       diff= 0.644851
left_shoulder_roll_joint:  diff= 0.430495
left_shoulder_pitch_joint: diff= 0.319717
```

Corresponding `q_des` examples:

```text
left_shoulder_roll:
  playback q_des = 0.437267
  recorded q_des = 0.407058

left_shoulder_pitch:
  playback q_des = 0.060458
  recorded q_des = 0.038023

waist_yaw:
  playback q_des = -0.061333
  recorded q_des = -0.077680
```

### Substeps 2 and 3

Playback continues using the stale `s0` `q_des`, while recording uses the updated `s1` `q_des`.

Substep 2:

```text
body q_des diff: 5.720666e-02
body ctrl diff:  1.151472e+00
post_qvel diff:  2.660704e-01
```

Substep 3:

```text
body q_des diff: 5.720666e-02
body ctrl diff:  1.098278e+00
post_qvel diff:  2.945817e-01
```

After Frame 12 step 4, compared to recorded Frame 13 s0:

```text
qpos full diff: 5.003188e-03
qvel full diff: 3.140109e-01
body qvel diff: 2.963508e-01
```

## Conclusion

The recording does not guarantee one constant `q_des` per 50 Hz frame. The simulator runs at 200 Hz and reads the latest asynchronous `LowCmd` at each `mj_step`.

For Frame 12 specifically:

```text
substep 0 uses old q_des
substeps 1-3 use new q_des
```

Playback currently uses one per-frame body target for all four live substeps, so it matches substep 0 but diverges from substep 1 onward.

The mismatch is not caused by `qacc_warmstart`. The exact differing parameter is `robot.motor_pd_substep_q_des`, which directly explains the recorded `ctrl` differences.
