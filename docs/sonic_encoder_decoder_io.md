# SONIC Encoder And Decoder I/O

## Raw ONNX Interfaces

| Model | Input | Input Shape | Output | Output Shape |
|---|---|---:|---|---:|
| `model_encoder.onnx` | `obs_dict` | `[1, 1762]` | `encoded_tokens` | `[1, 64]` |
| `model_decoder.onnx` | `obs_dict` | `[1, 994]` | `action` | `[1, 29]` |

## Encoder Input `[1762]`

Layout from `playback_lerobot.py` / `observation_config.yaml`:

| Range | Field |
|---:|---|
| `0:4` | `encoder_mode_4` |
| `4:294` | `motion_joint_positions_10frame_step5` |
| `294:584` | `motion_joint_velocities_10frame_step5` |
| `584:594` | `motion_root_z_position_10frame_step5` |
| `594:595` | `motion_root_z_position` |
| `595:601` | `motion_anchor_orientation` |
| `601:661` | `motion_anchor_orientation_10frame_step5` |
| `661:781` | `motion_joint_positions_lowerbody_10frame_step5` |
| `781:901` | `motion_joint_velocities_lowerbody_10frame_step5` |
| `901:910` | `vr_3point_local_target` |
| `910:922` | `vr_3point_local_orn_target` |
| `922:1642` | `smpl_joints_10frame_step1` |
| `1642:1702` | `smpl_anchor_orientation_10frame_step1` |
| `1702:1762` | `motion_joint_positions_wrists_10frame_step1` |

For current SMPL mode, the main active parts are:

| Field | Meaning |
|---|---|
| `encoder_mode_4` | Mode id `2` for SMPL |
| `smpl_joints_10frame_step1` | `10 x 72` |
| `smpl_anchor_orientation_10frame_step1` | `10 x 6` |
| `motion_joint_positions_wrists_10frame_step1` | `10 x 6` |

Encoder output:

| Output | Shape |
|---|---:|
| `encoded_tokens` / `token_state` | `[64]` |

## Decoder Input `[994]`

| Range | Field | Shape |
|---:|---|---:|
| `0:64` | `token_state` | `[64]` |
| `64:94` | `his_base_angular_velocity_10frame_step1` | `10 x 3` |
| `94:384` | `his_body_joint_positions_10frame_step1` | `10 x 29` |
| `384:674` | `his_body_joint_velocities_10frame_step1` | `10 x 29` |
| `674:964` | `his_last_actions_10frame_step1` | `10 x 29` |
| `964:994` | `his_gravity_dir_10frame_step1` | `10 x 3` |

Decoder output:

| Output | Shape |
|---|---:|
| `action` | `[29]` |
