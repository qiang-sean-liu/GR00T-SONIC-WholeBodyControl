# Episode 3 `robot.motor_pd_q_des` Analysis

- Dataset: `/home/horizon/wrk/SONIC/GR00T-WholeBodyControl/outputs/onnx-inputs-debug/data/chunk-000/episode_000003.parquet`
- Field: `robot.motor_pd_q_des`
- Shape: `1125 x 29` body joints, MuJoCo/body order

## What This Field Is

`robot.motor_pd_q_des` is the frame-level body position target read from the latest DDS `LowCmd` stored on the MuJoCo side. In `base_sim.py`, it is recorded from `low_cmd.motor_cmd[i].q` through `_motor_cmd_array("q", ...)`. The substep version, `robot.motor_pd_substep_q_des`, is the same quantity captured before each `mj_step`.

Important: it is not the policy raw action. It is the final body joint target after C++ policy post-processing: raw decoder action -> IsaacLab-to-MuJoCo remap -> `g1_action_scale` -> `default_angles` -> float cast -> `LowCmd.motor_cmd[i].q`.

## Completeness

- Finite values: `32625` / `32625`
- NaN values: `0`
- Frame indices: `0` to `1124`, contiguous: `True`

## Relation To Substep `q_des`

`robot.motor_pd_q_des` matches the latest valid substep `q_des` in `1125` / `1125` frames. The max L2 difference is `0.000e+00`.

| Substep compared to frame-level `motor_pd_q_des` | Exact matches | Compared frames | Mean L2 | Max L2 |
|---:|---:|---:|---:|---:|
| s0 | 201 | 1125 | 9.152e-02 | 1.542e+00 |
| s1 | 463 | 1125 | 6.652e-02 | 1.542e+00 |
| s2 | 832 | 1124 | 2.483e-02 | 6.433e-01 |
| s3 | 1124 | 1124 | 0.000e+00 | 0.000e+00 |

This means the frame-level value is effectively the newest command in the retained substep history, not necessarily the command used by all four substeps.

Substep uniqueness per 50 Hz frame:

| Distinct `q_des` vectors inside the 4 substeps | Frame count |
|---:|---:|
| 1 | 201 |
| 2 | 918 |
| 3 | 6 |

Frames with 3 distinct substep `q_des` values: `[91, 246, 298, 612, 922, 997]`. These are rare timing/overrun cases where the four MuJoCo steps span more than one C++ control-command boundary.

## Relation To `sonic.q_target_cmd`

The best alignment is usually same-frame `sonic.q_target_cmd`, because `robot.motor_pd_q_des` is the latest command visible to MuJoCo at the end of the frame-level substep window. However, due to DDS and sim-loop timing, some frames still align to previous or nearby policy outputs.

| Offset in `sonic.q_target_cmd[i + offset]` | Compared frames | Exact matches | Mean L2 | Max L2 |
|---:|---:|---:|---:|---:|
| -3 | 1122 | 0 | 2.370e-01 | 2.120e+00 |
| -2 | 1123 | 0 | 1.626e-01 | 1.745e+00 |
| -1 | 1124 | 417 | 7.070e-02 | 1.152e+00 |
| +0 | 1125 | 704 | 3.740e-02 | 1.542e+00 |
| +1 | 1124 | 1 | 1.372e-01 | 1.961e+00 |
| +2 | 1123 | 0 | 2.161e-01 | 2.232e+00 |
| +3 | 1122 | 0 | 2.808e-01 | 2.407e+00 |

Best-offset histogram per frame:

| Best `q_target_cmd` offset | Frame count |
|---:|---:|
| -2 | 10 |
| -1 | 419 |
| +0 | 696 |

## Frame-To-Frame Changes

- Frames where `robot.motor_pd_q_des` changes from previous frame: `1018` / `1124`
- Mean frame-to-frame L2 jump: `1.060e-01`
- Max frame-to-frame L2 jump: `1.542e+00`

Largest frame-to-frame jumps:

| Rank | Frame | Timestamp (s) | L2 from previous | Largest joint contributor | Joint delta | Best q_target offset |
|---:|---:|---:|---:|---|---:|---:|
| 1 | 485 | 9.700 | 1.542e+00 | `right_ankle_pitch` | -1.276e+00 | -1 |
| 2 | 431 | 8.620 | 1.152e+00 | `right_ankle_pitch` | -9.686e-01 | +0 |
| 3 | 28 | 0.560 | 9.364e-01 | `right_elbow` | -7.985e-01 | +0 |
| 4 | 69 | 1.380 | 8.616e-01 | `right_shoulder_pitch` | -4.863e-01 | +0 |
| 5 | 233 | 4.660 | 8.510e-01 | `left_ankle_pitch` | -3.660e-01 | +0 |
| 6 | 898 | 17.960 | 7.950e-01 | `right_shoulder_pitch` | -5.897e-01 | -1 |
| 7 | 673 | 13.460 | 7.552e-01 | `right_ankle_pitch` | +4.363e-01 | +0 |
| 8 | 488 | 9.760 | 7.405e-01 | `right_ankle_pitch` | +5.290e-01 | -1 |
| 9 | 489 | 9.780 | 6.972e-01 | `right_ankle_pitch` | +5.217e-01 | -1 |
| 10 | 430 | 8.600 | 6.945e-01 | `left_hip_roll` | +3.826e-01 | +0 |
| 11 | 713 | 14.260 | 6.433e-01 | `right_ankle_pitch` | +4.496e-01 | +0 |
| 12 | 667 | 13.340 | 6.095e-01 | `left_ankle_pitch` | +3.650e-01 | +0 |
| 13 | 708 | 14.160 | 6.029e-01 | `right_ankle_pitch` | -4.155e-01 | +0 |
| 14 | 707 | 14.140 | 5.997e-01 | `right_ankle_pitch` | -4.726e-01 | +0 |
| 15 | 467 | 9.340 | 5.659e-01 | `right_ankle_pitch` | +3.677e-01 | +0 |
| 16 | 412 | 8.240 | 5.656e-01 | `left_ankle_pitch` | -4.252e-01 | +0 |
| 17 | 712 | 14.240 | 5.512e-01 | `right_ankle_pitch` | +3.260e-01 | +0 |
| 18 | 111 | 2.220 | 5.442e-01 | `right_ankle_pitch` | -4.478e-01 | +0 |
| 19 | 194 | 3.880 | 5.305e-01 | `right_ankle_pitch` | -2.975e-01 | -1 |
| 20 | 60 | 1.200 | 5.267e-01 | `left_elbow` | -3.806e-01 | +0 |

## Per-Joint Motion Range

Top joints by range of `robot.motor_pd_q_des` across the episode:

| Rank | Joint index | Joint | Min q_des | Max q_des | Range |
|---:|---:|---|---:|---:|---:|
| 1 | 10 | `right_ankle_pitch` | -0.674772 | +1.485130 | 2.159903 |
| 2 | 22 | `right_shoulder_pitch` | -1.715428 | +0.339506 | 2.054933 |
| 3 | 4 | `left_ankle_pitch` | -0.494058 | +1.335635 | 1.829693 |
| 4 | 18 | `left_elbow` | -0.814526 | +0.911136 | 1.725662 |
| 5 | 25 | `right_elbow` | -0.394495 | +1.309466 | 1.703961 |
| 6 | 15 | `left_shoulder_pitch` | -0.771951 | +0.609684 | 1.381635 |
| 7 | 26 | `right_wrist_roll` | -0.171479 | +0.992346 | 1.163824 |
| 8 | 23 | `right_shoulder_roll` | -1.395192 | -0.239807 | 1.155385 |
| 9 | 9 | `right_knee` | -0.056741 | +1.060000 | 1.116741 |
| 10 | 3 | `left_knee` | +0.103442 | +1.169432 | 1.065990 |

## Stale ZMQ Frames

There are `22` frames where `observation.state`, `sonic.decoder_obs`, and `sonic.decoder_action_raw` are identical to the previous frame. Those frames are:

`157, 162, 166, 171, 177, 181, 186, 216, 221, 226, 231, 471, 476, 481, 486, 491, 496, 805, 810, 815, 820, 825`

Among those stale frames, `robot.motor_pd_q_des` changed in `21` frames:

`157, 162, 166, 171, 177, 181, 186, 216, 221, 226, 471, 476, 481, 486, 491, 496, 805, 810, 815, 820, 825`

This is useful diagnostically: the policy/debug stream can repeat while the MuJoCo LowCmd-derived command field still advances, because they come through different timing paths.

## Notable Interpretation

- `robot.motor_pd_q_des` is the command that MuJoCo most recently saw at the frame/debug-snapshot level.
- It should not be assumed to be constant over all 4 substeps; use `robot.motor_pd_substep_q_des` when exact replay of each `mj_step` input is required.
- It usually corresponds to `sonic.q_target_cmd` from the same frame, but the exact alignment can shift by one or more frames when the 200 Hz MuJoCo loop, 50 Hz policy loop, 500 Hz command writer, and exporter sampling are not phase locked.
- For playback fidelity, `robot.motor_pd_substep_q_des` is the correct source for substep replay; `robot.motor_pd_q_des` is useful as the latest-command summary for that dataset row.

## Appendix: Rare 3-Command Substep Frames

| Frame | Substep sim times | Substep wall span (ms) | Substep-to-q_target mapping |
|---:|---|---:|---|
| 91 | `17.005, 17.010, 17.015, 17.020` | 23.12 | s0->q_target[89] (L2 0.0e+00); s1->q_target[90] (L2 0.0e+00); s2->q_target[90] (L2 0.0e+00); s3->q_target[91] (L2 0.0e+00) |
| 246 | `19.905, 19.910, 19.915, 19.920` | 21.25 | s0->q_target[244] (L2 0.0e+00); s1->q_target[245] (L2 0.0e+00); s2->q_target[245] (L2 0.0e+00); s3->q_target[246] (L2 0.0e+00) |
| 298 | `20.855, 20.860, 20.865, 20.870` | 27.94 | s0->q_target[296] (L2 0.0e+00); s1->q_target[297] (L2 0.0e+00); s2->q_target[297] (L2 0.0e+00); s3->q_target[298] (L2 0.0e+00) |
| 612 | `26.710, 26.715, 26.720, 26.725` | 19.57 | s0->q_target[610] (L2 0.0e+00); s1->q_target[611] (L2 0.0e+00); s2->q_target[611] (L2 0.0e+00); s3->q_target[612] (L2 0.0e+00) |
| 922 | `32.525, 32.530, 32.535, 32.540` | 23.16 | s0->q_target[920] (L2 0.0e+00); s1->q_target[921] (L2 0.0e+00); s2->q_target[921] (L2 0.0e+00); s3->q_target[922] (L2 0.0e+00) |
| 997 | `33.925, 33.930, 33.935, 33.940` | 22.84 | s0->q_target[995] (L2 0.0e+00); s1->q_target[996] (L2 0.0e+00); s2->q_target[996] (L2 0.0e+00); s3->q_target[997] (L2 0.0e+00) |

