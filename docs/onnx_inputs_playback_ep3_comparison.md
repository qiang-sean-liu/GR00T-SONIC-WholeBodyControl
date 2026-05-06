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
Frame 2:  L2_q_target=6.114e-07  L2_state=2.462e-03
Frame 3:  L2_q_target=4.586e-07  L2_state=4.286e-03
Frame 4:  L2_q_target=5.886e-07  L2_state=4.250e-03
Frame 11: L2_q_target=7.371e-07  L2_state=4.659e-03
```

Analysis:

- Encoder output (`token`) is exactly the same as recording, same as original playback.
- During the first 10 playback frames, decoder targets match the recording to around `1e-6` because the decoder receives recorded `sonic.decoder_obs`.
- During the same first 10 frames, MuJoCo state follows recorded substep replay instead of recomputed policy physics.
- After the warmup window, playback switches back to live closed-loop decoder history and free-base MuJoCo physics, so target and state differences reappear.
- This is a valid change if the goal is to remove the artificial one-frame-history initialization error at the beginning. It is not meant to make the whole episode exactly match recording.

## Direct Comparison

The warmup version improves the beginning of the episode:

```text
Frame 2 q_target L2:
original = 7.627e-02
warmup   = 6.114e-07

Frame 11 q_target L2:
original = 5.757e-02
warmup   = 7.371e-07
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
