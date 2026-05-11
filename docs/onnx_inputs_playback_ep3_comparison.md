# ONNX Inputs Playback Episode 3 Comparison

Dataset:

```bash
outputs/onnx-inputs-debug/data/chunk-000/episode_000003.parquet
```

Environment:

```bash
env_name=default
camera=head_camera
fps=50
free_base=true
init_frame=f1s0
start_frame=1
```

Note: `L2_state` in these scripts compares simulated body joint state after a playback step against `observation.state` at the next frame. `observation.state` comes from the async debug stream, so it is useful for trend comparison but is not an exact MuJoCo substep equality check.

## Original Closed-Loop Playback

This is the original ONNX-input playback. It initializes from one frame, uses recorded `sonic.encoder_obs`, then builds decoder history from the live playback MuJoCo state starting immediately at frame 1.

```bash
MUJOCO_GL=egl ./.venv_sim/bin/python gear_sonic/scripts/playback_lerobot_onnx_inputs_pd.py \
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
  --compare \
  --no_viewer \
  --camera head_camera \
  --output_video outputs/onnx-inputs-debug/videos/onnx_playback_compare/episode_000003_original_closed_loop.mp4
```

Video:

```bash
outputs/onnx-inputs-debug/videos/onnx_playback_compare/episode_000003_original_closed_loop.mp4
```

Summary:

```text
token:               mean=0.000e+00  median=0.000e+00  max=0.000e+00
policy_target_delta: mean=1.915e-01  median=1.259e-01  max=2.309e+00
q_target:            mean=1.915e-01  median=1.259e-01  max=2.309e+00
sim_body_state:      mean=8.391e-02  median=7.516e-02  max=2.700e-01
```

Early frames:

```text
Frame 2:  L2_q_target=7.627e-02  L2_state=2.579e-03
Frame 3:  L2_q_target=1.012e-01  L2_state=6.220e-03
Frame 4:  L2_q_target=9.727e-02  L2_state=9.492e-03
Frame 11: L2_q_target=5.757e-02  L2_state=1.230e-02
```

Analysis:

- Encoder output (`token`) is exactly the same as recording because playback feeds recorded `sonic.encoder_obs`.
- Decoder output differs immediately because decoder history is generated from the playback state/history, not the recorded deploy history.
- This is the closest mode to testing closed-loop ONNX policy rollout from the first playback frame.

## Decoder And Physics Warmup Playback

This variant uses recorded `sonic.decoder_obs` for the first 10 playback frames and replays recorded MuJoCo substeps for the first 10 playback frames. After that, it switches back to closed-loop decoder history and closed-loop physics.

```bash
MUJOCO_GL=egl ./.venv_sim/bin/python gear_sonic/scripts/playback_lerobot_onnx_inputs_pd_decoder_warmup.py \
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
  --compare \
  --no_viewer \
  --camera head_camera \
  --output_video outputs/onnx-inputs-debug/videos/onnx_playback_compare/episode_000003_decoder_physics_warmup.mp4
```

Video:

```bash
outputs/onnx-inputs-debug/videos/onnx_playback_compare/episode_000003_decoder_physics_warmup.mp4
```

Summary:

```text
token:          mean=0.000e+00  median=0.000e+00  max=0.000e+00
q_target:       mean=2.041e-01  median=1.290e-01  max=2.330e+00
sim_body_state: mean=8.116e-02  median=6.070e-02  max=3.048e-01
```

Early frames:

```text
Frame 2:  L2_q_target=5.987e-07  L2_state=2.462e-03
Frame 3:  L2_q_target=4.592e-07  L2_state=4.286e-03
Frame 4:  L2_q_target=5.962e-07  L2_state=4.250e-03
Frame 11: L2_q_target=7.471e-07  L2_state=4.659e-03
```

Analysis:

- Encoder output (`token`) is exactly the same as recording, same as original playback.
- During the first 10 playback frames, decoder targets match the recording to around `1e-6` because the decoder receives recorded `sonic.decoder_obs`.
- During the same first 10 frames, MuJoCo state follows recorded substep replay instead of recomputed policy physics.
- After the warmup window, playback switches back to live closed-loop decoder history and free-base MuJoCo physics, so target and state differences reappear.
- This is a valid change if the goal is to remove the artificial one-frame-history initialization error at the beginning. It is not meant to make the whole episode exactly match recording.

The small first-frame target difference is deterministic. Repeating the warmup first-frame decoder path 100 times produced the same value every run:

```text
L2_q_target min  = 5.986970970413e-07
L2_q_target max  = 5.986970970413e-07
L2_q_target mean = 5.986970970413e-07
L2_q_target std  = 0.0
target_max_abs_diff_across_runs = 0.0
```

The decoder input itself is not missing or different. For frame 2 (`i=1`), playback passes the recorded `sonic.decoder_obs` cast to float32, and this equals the recorded tensor cast to float32 exactly:

```text
decoder_obs_rec dtype loaded: float64
decoder input passed to ONNX: float32
decoder input equals recorded cast to float32: True
decoder input shape: (994,)
```

The remaining tiny delta comes from rerunning the decoder in a different inference backend than teleop deploy. The recording metadata for `outputs/onnx-inputs-debug` says:

```text
model_path: policy/release/model_decoder.onnx
policy_fp16: false
```

The C++ deploy path initializes the control policy through `PolicyEngine`, which is the TensorRT control policy path. Therefore the recorded `sonic.decoder_action_raw` / `sonic.q_target_cmd` came from:

```text
C++ deploy -> PolicyEngine -> TensorRT -> FP32
```

The Python playback path uses ONNXRuntime. On this machine it ran with CPU only:

```text
Available providers: AzureExecutionProvider, CPUExecutionProvider
providers: ['CPUExecutionProvider']
```

So the exact backend difference for this comparison is:

```text
recording/teleop: C++ TensorRT FP32
playback:         Python ONNXRuntime CPU FP32
```

With the same recorded decoder input, the decoder output comparison at frame 2 was:

```text
raw decoder output playback vs recorded sonic.decoder_action_raw:
  raw_l2=1.662715e-06
  raw_max_abs=9.536743e-07

q_target playback vs recorded sonic.q_target_cmd:
  l2=5.986971e-07
  max_abs=4.172325e-07
```

For exactly zero target difference, playback must use the recorded `sonic.decoder_action_raw` or the final recorded `sonic.q_target_cmd` directly instead of rerunning the decoder model.

### C++ TensorRT Replay Proof

To prove the backend explanation without changing the live teleop implementation, an isolated test pipeline was added:

```text
gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/tests/replay_policy_engine_decoder_test.cpp
gear_sonic/scripts/compare_recorded_decoder_with_policy_engine.py
```

The C++ executable is a separate target, `replay_policy_engine_decoder_test`. It does not link into or modify `g1_deploy_onnx_ref`; it only reuses the same `PolicyEngine` TensorRT path on recorded `sonic.decoder_obs` tensors prepared by the Python wrapper.

Build and run:

```bash
cmake --build gear_sonic_deploy/build --target replay_policy_engine_decoder_test -j2

./.venv_sim/bin/python gear_sonic/scripts/compare_recorded_decoder_with_policy_engine.py \
  --dataset_dir outputs/onnx-inputs-debug \
  --episode 3 \
  --model gear_sonic_deploy/policy/release/model_decoder.onnx \
  --executable gear_sonic_deploy/target/release/replay_policy_engine_decoder_test
```

Full-episode result for episode 3:

```text
PolicyEngine decoder replay test
  precision=FP32
  frames=1125

raw_action_vs_recorded:
  mean_l2=0
  max_l2=0
  mean_max_abs=0
  max_abs=0

q_target_vs_recorded:
  mean_l2=0
  max_l2=0
  mean_max_abs=0
  max_abs=0
```

This proves that recorded `sonic.decoder_obs` fed through the C++ TensorRT FP32 `PolicyEngine` reproduces recorded `sonic.decoder_action_raw` and `sonic.q_target_cmd` exactly. Therefore the `~6.1e-7` warmup playback difference is not caused by missing decoder inputs.

### Is The Python Playback Implementation Incorrect?

The main Python behavior is not incorrect for ONNXRuntime playback: it intentionally reruns `model_decoder.onnx` through Python ONNXRuntime, and this backend is not bit-identical to C++ TensorRT. That backend difference is the reason recorded decoder inputs still produce tiny non-zero differences when the decoder is rerun in Python.

One small Python-side post-processing mismatch was found during the double-check: C++ teleop stores `MotorCommand::q_target` as `float`, while Python previously kept the computed q target in float64 after `default_angles + raw_action * action_scale`. With recorded raw actions, this caused only a very small post-processing delta:

```text
python_current_float64:
  max_l2=9.404486463151749e-08
  mean_l2=5.1864529509206476e-08
  max_abs=5.943128078556015e-08

cxx_like_float_store:
  first frame diff = 0 for first 8 joints
```

That post-processing mismatch is smaller than the observed ONNXRuntime-vs-TensorRT decoder delta and is not the main cause of the frame-2 q-target difference. After correcting the cast, the same frame still has `L2_q_target=5.986971e-07`, while `raw_l2=1.662715e-06` is unchanged. `gear_sonic/scripts/playback_lerobot.py` now casts computed body q targets through float32 before returning them, matching the C++ `static_cast<float>(default_angles[i] + action_value)` behavior more closely.

## Direct Comparison

The warmup version improves the beginning of the episode:

```text
Frame 2 q_target L2:
original = 7.627e-02
warmup   = 5.987e-07

Frame 11 q_target L2:
original = 5.757e-02
warmup   = 7.471e-07
```

Whole-episode state difference is similar:

```text
sim_body_state mean:
original = 8.391e-02
warmup   = 8.116e-02

sim_body_state median:
original = 7.516e-02
warmup   = 6.070e-02
```

Conclusion:

- Use `playback_lerobot_onnx_inputs_pd.py` to test pure closed-loop policy playback from frame 1.
- Use `playback_lerobot_onnx_inputs_pd_decoder_warmup.py` to start from recorded decoder and MuJoCo history for 10 frames, then test closed-loop rollout after the history buffer is populated.

## PD-Action Substep Playback

This command uses `playback_lerobot_action_pd.py` instead of the ONNX decoder scripts. It replays the recorded MuJoCo substep inputs directly, so it does not use `--sonic_encoder`, `--sonic_decoder`, or decoder warmup flags.

```bash
./.venv_sim/bin/python gear_sonic/scripts/playback_lerobot_action_pd.py \
  --dataset_dir outputs/onnx-inputs-debug \
  --episode 3 \
  --env_name default \
  --fps 50 \
  --free_base \
  --init_substep_frame 1 \
  --init_substep_index 0 \
  --init_substep_all_inputs \
  --start_frame 1 \
  --use_recorded_substep_inputs
```

## PD Gain Source

During C++ ONNX teleop/deploy, `kp` and `kd` are not loaded from YAML. They come from constants in `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp`:

```text
const std::array<float, 29> kps = { ... };
const std::array<float, 29> kds = { ... };
```

`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` copies those constants into every policy `MotorCommand`:

```text
motor_command_tmp.kp.at(i) = kps[i];
motor_command_tmp.kd.at(i) = kds[i];
```

So if `policy_parameters.hpp` is not changed, each joint's `kp/kd` stays constant over time. They can differ between joints, but the same joint receives the same `kp/kd` every normal ONNX policy command. In that same mode, `dq_target` is `0.0`, `tau_ff` is `0.0`, `q_target` changes with policy action, and the final PD torque still changes every MuJoCo substep because the measured `q/dq` state changes.

## Teleop `action.wbc` To `LowCmd`

In live C++ ONNX teleop, the policy output is converted to a body position command in `CreatePolicyCommand()`:

```text
action_value = decoder_action_raw[isaaclab_to_mujoco[i]] * g1_action_scale[i]
q_target[i] = default_angles[i] + action_value
tau_ff[i] = 0.0
kp[i] = kps[i]
kd[i] = kds[i]
dq_target[i] = 0.0
```

`g1_action_scale[i]` is per-joint, not one shared constant. It is defined in `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp` as:

```text
action_scale = 0.25 * effort_limit / stiffness
```

Both `effort_limit` and `stiffness` are code constants, so the resulting scale is constant during runtime for a given joint. They are not learned or updated during teleop. The stiffness values are computed from fixed armature constants:

```text
NATURAL_FREQ = 10 * 2*pi
STIFFNESS_5020    = ARMATURE_5020    * NATURAL_FREQ^2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ^2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ^2
STIFFNESS_4010    = ARMATURE_4010    * NATURAL_FREQ^2
```

The effort limits are also fixed constants:

```text
EFFORT_LIMIT_5020    = 25.0
EFFORT_LIMIT_7520_14 = 88.0
EFFORT_LIMIT_7520_22 = 139.0
EFFORT_LIMIT_4010    = 5.0
```

Example from the code: `left_hip_pitch_joint` uses the `7520_22` constants:

```text
g1_action_scale[0] = 0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22
                   = 0.350661466
```

The unique scale values are:

```text
0.350661466
0.547546465
0.438577314
0.074500870
```

Examples:

```text
hip_pitch / hip_roll / knee: 0.350661466
hip_yaw / waist_yaw:         0.547546465
ankle / shoulder / elbow:    0.438577314
wrist_pitch / wrist_yaw:     0.074500870
```

So each joint can scale the raw decoder action differently before adding `default_angles[i]`.

That `MotorCommand` is then stored in `motor_command_buffer_`. The 500 Hz command-writer thread reads the latest `MotorCommand` and packs it into `LowCmd_`:

```text
LowCmd.motor_cmd[i].q = MotorCommand.q_target[i]
LowCmd.motor_cmd[i].dq = MotorCommand.dq_target[i]
LowCmd.motor_cmd[i].tau = MotorCommand.tau_ff[i]
LowCmd.motor_cmd[i].kp = MotorCommand.kp[i]
LowCmd.motor_cmd[i].kd = MotorCommand.kd[i]
```

The dataset's `action.wbc` is recorded by the Python exporter from the C++ debug field `last_action`, then expanded through the robot model into the 43-joint dataset action layout. In this debug stream, `last_action` is already serialized as a MuJoCo-order q target:

```text
last_action_mujoco[i] = state.last_action[isaaclab_to_mujoco[i]] * g1_action_scale[i] + default_angles[i]
```

`sonic.q_target_cmd` and `robot.motor_q` are the direct recorded final body q command fields, sourced from `q_target_cmd` / `body_motor_cmd_q` / `motor_q`.

Rechecked dataset:

```bash
/home/horizon/wrk/SONIC/GR00T-WholeBodyControl/outputs/onnx-inputs-debug/data/chunk-000/episode_000003.parquet
```

This episode has 1125 rows and contains `action.wbc`, `sonic.q_target_cmd`, and `robot.motor_q`. The data shows `action.wbc` is **not** the next-frame lowcmd q. The best alignment is previous-frame lowcmd q:

```text
action.wbc[t] body vs sonic.q_target_cmd[t-1]:
  rows=1124
  mean_l2=5.155e-03
  median_l2=5.036e-08
  max_l2=4.385e-01
  rows_allclose_1e-6=1077/1124

action.wbc[t] body vs sonic.q_target_cmd[t]:
  rows=1125
  mean_l2=1.076e-01
  median_l2=7.177e-02
  max_l2=1.542e+00
  rows_allclose_1e-6=0/1125

action.wbc[t] body vs sonic.q_target_cmd[t+1]:
  rows=1124
  mean_l2=1.907e-01
  median_l2=1.276e-01
  max_l2=1.961e+00
  rows_allclose_1e-6=0/1124
```

The same numbers hold for `robot.motor_q`, because `robot.motor_q` and `sonic.q_target_cmd` are identical in this dataset. A direct sample also shows the offset:

```text
action.wbc[1] body first 8 =
[-0.061858, 0.029379, 0.100620, 0.242781, 0.280341, 0.004069, 0.025464, -0.055602]

sonic.q_target_cmd[0] first 8 =
[-0.061858, 0.029379, 0.100620, 0.242781, 0.280341, 0.004069, 0.025464, -0.055602]
```

Interpretation: the exporter samples asynchronous debug fields at 50 Hz. `action.wbc[t]` is mostly the previous policy q target, approximately `lowcmd_q[t-1]`, while `sonic.q_target_cmd` / `robot.motor_q` are the direct command fields for the sampled debug message. For command-accurate replay, prefer `sonic.q_target_cmd` or `robot.motor_q` over deriving body q from `action.wbc`.

## MuJoCo Restore Fields And `qacc_warmstart`

Pure substep playback restores these MuJoCo fields before each `mj_step`:

```text
data.time
data.ctrl
data.qfrc_applied
data.xfrc_applied
data.qacc_warmstart
```

For episode 3 from `outputs/onnx-inputs-debug` and episode 9 from `outputs/2026-05-05-17-47-00-onnx-inputs-debug`, these fields are not all the same:

```text
data.time:           not same
data.ctrl:           not same
data.qfrc_applied:   same, all zeros
data.xfrc_applied:   same, all zeros
data.qacc_warmstart: not same
```

`qacc_warmstart` is not produced by teleop or by the policy. It is an internal MuJoCo solver state used as the initial guess for the next constraint/dynamics solve. During teleop, the code computes PD torque from `LowCmd`, writes it into `data.ctrl`, records the pre-step `qacc_warmstart`, then calls `mj_step`. MuJoCo updates `qacc_warmstart` internally as part of the physics solve.

Measured `qacc_warmstart` difference between those two episodes:

```text
episode 3 range: min=-2646.94  max=3101.12
episode 9 range: min=-1596.68  max=1418.79

cross comparison:
  mean_l2=446.62
  median_l2=356.75
  max_l2=4309.39
  max_abs=3100.25
```

This difference means the two episodes are on different MuJoCo trajectories/contact/solver histories. `kp/kd` can be identical while `qacc_warmstart` differs, because `qacc_warmstart` depends on the evolving `qpos`, `qvel`, `ctrl`, contacts, constraints, and prior solver state.

For the same comparison, substep `kp/kd` are identical:

```text
robot.motor_pd_substep_kp: same exactly
robot.motor_pd_substep_kd: same exactly
```

Episode 9 has frame-level `robot.motor_kp` / `robot.motor_kd` as `NaN`, but its substep-level `robot.motor_pd_substep_kp/kd` are fully recorded and match episode 3 exactly.

## Future-Input Encoder Playback

This variant rebuilds the encoder future inputs from the recorded teleop future fields instead of directly feeding the full recorded `sonic.encoder_obs`. It then uses the same closed-loop decoder and physics rollout as the original ONNX-input playback.

```bash
./.venv_sim/bin/python gear_sonic/scripts/playback_lerobot_onnx_future_inputs_pd.py \
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
  --compare \
  --no_viewer
```

The encoder future window is 10 continuous frames:

```text
t, t+1, t+2, ..., t+9
```

It is not downsampled. The encoder layout names are `smpl_joints_10frame_step1`, `smpl_anchor_orientation_10frame_step1`, and `motion_joint_positions_wrists_10frame_step1`; `step1` means adjacent frames.

The script reconstructs these recorded future sections:

```text
smpl_joints_10frame_step1              <- teleop.smpl_joints[t:t+10]
motion_joint_positions_wrists_step1    <- concat(teleop.left_wrist_joints, teleop.right_wrist_joints)[t:t+10]
```

When fewer than 10 future frames remain, it falls back to the full recorded `sonic.encoder_obs`. For episode 3 with `start_frame=1`, that produced:

```text
Encoder source counts: reconstructed_future=1115, recorded_fallback=9
```

Encoder reconstruction check against recorded `sonic.encoder_obs`:

```text
valid_future_windows: 1116
fallback_frames:      9

all_l2_mean:          0.0
all_l2_median:        0.0
all_l2_max:           0.0
all_max_abs:          0.0
exact_equal_float32:  True

smpl_joints_10frame_step1:
  l2_mean=0.0  l2_max=0.0  max_abs=0.0  exact=True

motion_joint_positions_wrists_10frame_step1:
  l2_mean=0.0  l2_max=0.0  max_abs=0.0  exact=True
```

Playback summary:

```text
encoder_obs:         mean=9.081e-08  median=3.441e-08  max=5.178e-07
token:               mean=0.000e+00  median=0.000e+00  max=0.000e+00
policy_target_delta: mean=1.915e-01  median=1.259e-01  max=2.309e+00
q_target:            mean=1.915e-01  median=1.259e-01  max=2.309e+00
sim_body_state:      mean=8.391e-02  median=7.516e-02  max=2.700e-01
```

The `encoder_obs` playback summary is nonzero only because the script compares the full runtime encoder vector after float conversions and fallback handling. The direct future-section comparison above confirms the recorded future-frame sections are exactly equal to the corresponding sections of `sonic.encoder_obs` in float32.

Conclusion:

- The recorded future 10-frame teleop fields exactly reproduce the future sections of `sonic.encoder_obs` for all frames with a full future horizon.
- The encoder token remains exactly equal to the recording.
- Differences in q-target and simulated state are therefore not caused by encoder future input reconstruction; they still come from closed-loop decoder history and physics rollout.

## Episode 9 Playback Commands

Recorded video:

```bash
outputs/2026-05-05-17-47-00-onnx-inputs-debug/videos/chunk-000/observation.images.ego_view/episode_000009.mp4
```

Without warmup:

```bash
./.venv_sim/bin/python gear_sonic/scripts/playback_lerobot_onnx_inputs_pd.py \
  --dataset_dir outputs/2026-05-05-17-47-00-onnx-inputs-debug \
  --episode 9 \
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

With decoder/physics warmup:

```bash
./.venv_sim/bin/python gear_sonic/scripts/playback_lerobot_onnx_inputs_pd_decoder_warmup.py \
  --dataset_dir outputs/2026-05-05-17-47-00-onnx-inputs-debug \
  --episode 9 \
  --env_name default \
  --sonic_encoder gear_sonic_deploy/policy/release/model_encoder.onnx \
  --sonic_decoder gear_sonic_deploy/policy/release/model_decoder.onnx \
  --fps 50 \
  --free_base \
  --init_substep_frame 1 \
  --init_substep_index 0 \
  --init_substep_all_inputs \
  --start_frame 1 \
  --recorded_decoder_warmup_frames 10 \
  --recorded_physics_warmup_frames 10 \
  --compare
```

Add `--no_viewer` to either command for headless playback.
