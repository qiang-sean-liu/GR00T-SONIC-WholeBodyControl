"""Replay a LeRobot episode in MuJoCo — with optional SONIC encoder+policy inference.

Kinematic mode (default):
    Replays observation.state from a LeRobot Parquet file, bypassing PD control.

SONIC mode (--sonic_encoder / --sonic_decoder):
    Runs the SONIC encoder (SMPL mode, no kinematic planner) and policy decoder
    on each frame.  The policy output replaces observation.state for the 29-DOF
    body joints; hand joints are always taken from the recorded action.
    Requires a v2.2 dataset (pico.smpl_joints, pico.body_root_quat columns).

43-DOF layout (matches convert_sonic_to_lerobot.py):
    state[0:6]   left_leg      (6)
    state[6:12]  right_leg     (6)
    state[12:15] waist         (3)
    state[15:22] left_arm      (7)
    state[22:29] left_hand     (7)
    state[29:36] right_arm     (7)
    state[36:43] right_hand    (7)

Usage:
    # Kinematic playback — overview camera:
    conda run -n sonic_dc python gear_sonic/scripts/playback_lerobot.py \\
        --dataset_dir /home/horizon/wrk/SONIC/lerobot_dataset_2.2 \\
        --episode 0 \\
        --env_name pnp_cube \\
        --output_video playback_ep0.mp4

    # SONIC inference playback — stereo ego + overview:
    conda run -n sonic_dc python gear_sonic/scripts/playback_lerobot.py \\
        --dataset_dir /home/horizon/wrk/SONIC/lerobot_dataset_2.2 \\
        --episode 0 \\
        --env_name pnp_cube \\
        --sonic_encoder gear_sonic_deploy/policy/release/model_encoder.onnx \\
        --sonic_decoder gear_sonic_deploy/policy/release/model_decoder.onnx \\
        --cameras overview head_camera_left head_camera_right \\
        --output_video playback_sonic_ep0.mp4 \\
        --compare

Environment → XML mapping:
    kitchen_pnp_apple  decoupled_wbc/control/robot_model/model_data/g1/kitchen_pnp_apple_43dof.xml
    pnp_cube           decoupled_wbc/control/robot_model/model_data/g1/pnp_cube_43dof.xml
    lift_box           decoupled_wbc/control/robot_model/model_data/g1/lift_box_43dof.xml
    pnp_bottle         decoupled_wbc/control/robot_model/model_data/g1/pnp_bottle_43dof.xml
    default            decoupled_wbc/control/robot_model/model_data/g1/scene_43dof.xml
"""

import argparse
import json
import math
import os
import pathlib
import time
from collections import deque
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_ENV_XML = {
    "kitchen_pnp_apple": "decoupled_wbc/control/robot_model/model_data/g1/kitchen_pnp_apple_43dof.xml",
    "pnp_cube":          "decoupled_wbc/control/robot_model/model_data/g1/pnp_cube_43dof.xml",
    "lift_box":          "decoupled_wbc/control/robot_model/model_data/g1/lift_box_43dof.xml",
    "pnp_bottle":        "decoupled_wbc/control/robot_model/model_data/g1/pnp_bottle_43dof.xml",
    "default":           "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml",
}

# 43-DOF slice → body (29) de-assembly
_BODY_IDX = np.concatenate([np.arange(22), np.arange(29, 36)])  # 29 joints
_LEFT_HAND_IDX = np.arange(22, 29)   # 7 joints
_RIGHT_HAND_IDX = np.arange(36, 43)  # 7 joints
_HAND_JOINT_SUFFIXES = [
    "index_0_joint",
    "index_1_joint",
    "middle_0_joint",
    "middle_1_joint",
    "thumb_0_joint",
    "thumb_1_joint",
    "thumb_2_joint",
]
# Upper-body body joints in 43-DOF layout: waist(12:15), left_arm(15:22), right_arm(29:36)
_WAIST_IDX = np.arange(12, 15)
_LEFT_ARM_IDX = np.arange(15, 22)
_RIGHT_ARM_IDX = np.arange(29, 36)

# Joint name substrings that identify body / hand joints (matches base_sim.py)
_BODY_JOINT_KEYS = ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]

# Default root pose: pelvis height for a standing robot (metres)
_ROOT_HEIGHT = 0.80

# ---------------------------------------------------------------------------
# SONIC model constants  (mirrors policy_parameters.hpp)
# ---------------------------------------------------------------------------

_NATURAL_FREQ = 10 * 2 * math.pi          # rad/s
_ARMATURE = {
    "5020":    0.003609725,
    "7520_14": 0.010177520,
    "7520_22": 0.025101925,
    "4010":    0.00425,
}
_EFFORT = {
    "5020":    25.0,
    "7520_14": 88.0,
    "7520_22": 139.0,
    "4010":    5.0,
}

def _action_scale(motor):
    k = _ARMATURE[motor] * _NATURAL_FREQ ** 2
    return 0.25 * _EFFORT[motor] / k

# Physics replay constants — from policy_parameters.hpp (gear_sonic_deploy).
# These are the ACTUAL gains sent by the C++ SONIC deploy code via DDS and applied
# by gear_sonic's base_sim.py compute_body_torques(). They differ substantially from
# the g1_29dof_sonic_model12.yaml MOTOR_KP values (which are standalone-sim defaults).
#
# Derived from:
#   NATURAL_FREQ = 10 * 2π ≈ 62.832 rad/s
#   STIFFNESS_xyz = ARMATURE_xyz * NATURAL_FREQ²
#   DAMPING_xyz   = 2 * DAMPING_RATIO(2) * ARMATURE_xyz * NATURAL_FREQ
#   = 4 * ARMATURE_xyz * NATURAL_FREQ
#
# MuJoCo body29 order: left_leg(6) right_leg(6) waist(3) left_arm(7) right_arm(7)
_SIM_DT               = 0.005   # matches SIMULATE_DT: 0.005 in g1_29dof_sonic_model12.yaml
_SIM_STEPS_PER_POLICY = 10      # 200 Hz sim / 20 Hz policy

# Use the same formula as policy_parameters.hpp to keep values in sync
_ω  = 10 * 2 * math.pi      # NATURAL_FREQ = 10 Hz
_ζ  = 2.0                   # DAMPING_RATIO (overdamped)

def _kp(motor: str) -> float:
    return _ARMATURE[motor] * _ω ** 2

def _kd(motor: str) -> float:
    return 2.0 * _ζ * _ARMATURE[motor] * _ω   # = 4 * ARMATURE * ω

_KP_BODY29 = np.array([
    _kp("7520_22"), _kp("7520_22"), _kp("7520_14"), _kp("7520_22"), 2*_kp("5020"), 2*_kp("5020"),  # left_leg
    _kp("7520_22"), _kp("7520_22"), _kp("7520_14"), _kp("7520_22"), 2*_kp("5020"), 2*_kp("5020"),  # right_leg
    _kp("7520_14"), 2*_kp("5020"), 2*_kp("5020"),                                                   # waist
    _kp("5020"), _kp("5020"), _kp("5020"), _kp("5020"), _kp("5020"), _kp("4010"), _kp("4010"),      # left_arm
    _kp("5020"), _kp("5020"), _kp("5020"), _kp("5020"), _kp("5020"), _kp("4010"), _kp("4010"),      # right_arm
], dtype=np.float64)

_KD_BODY29 = np.array([
    _kd("7520_22"), _kd("7520_22"), _kd("7520_14"), _kd("7520_22"), 2*_kd("5020"), 2*_kd("5020"),  # left_leg
    _kd("7520_22"), _kd("7520_22"), _kd("7520_14"), _kd("7520_22"), 2*_kd("5020"), 2*_kd("5020"),  # right_leg
    _kd("7520_14"), 2*_kd("5020"), 2*_kd("5020"),                                                   # waist
    _kd("5020"), _kd("5020"), _kd("5020"), _kd("5020"), _kd("5020"), _kd("4010"), _kd("4010"),      # left_arm
    _kd("5020"), _kd("5020"), _kd("5020"), _kd("5020"), _kd("5020"), _kd("4010"), _kd("4010"),      # right_arm
], dtype=np.float64)

# Torque limits from EFFORT_LIMIT constants in policy_parameters.hpp.
# hip_pitch/roll use 7520_22 motors (139 Nm); knee also 7520_22 (139 Nm).
# 2×5020 ankle/waist joints have 2×25=50 Nm.
_TORQUE_LIMIT_BODY29 = np.array([
    139.,  139.,  88., 139.,  50.,  50.,  # left_leg:  hip_pitch(139), hip_roll(139), hip_yaw(88), knee(139)
    139.,  139.,  88., 139.,  50.,  50.,  # right_leg
     88.,   50.,  50.,                    # waist:     waist_yaw(88), waist_roll(50), waist_pitch(50)
     25.,   25.,  25.,  25.,  25.,   5.,   5.,  # left_arm
     25.,   25.,  25.,  25.,  25.,   5.,   5.,  # right_arm
], dtype=np.float64)

# Hand PD gains — from HandCommandSender in gear_sonic
_KP_HAND          = 2.0
_KD_HAND          = 0.5
_TORQUE_LIMIT_HAND = 2.45   # from motor_effort_limit_list (first hand motor limit)

# 29-DOF action scale in MuJoCo body joint order
_ACTION_SCALE = np.array([
    _action_scale("7520_22"),  # left_hip_pitch
    _action_scale("7520_22"),  # left_hip_roll
    _action_scale("7520_14"),  # left_hip_yaw
    _action_scale("7520_22"),  # left_knee
    _action_scale("5020"),     # left_ankle_pitch
    _action_scale("5020"),     # left_ankle_roll
    _action_scale("7520_22"),  # right_hip_pitch
    _action_scale("7520_22"),  # right_hip_roll
    _action_scale("7520_14"),  # right_hip_yaw
    _action_scale("7520_22"),  # right_knee
    _action_scale("5020"),     # right_ankle_pitch
    _action_scale("5020"),     # right_ankle_roll
    _action_scale("7520_14"),  # waist_yaw
    _action_scale("5020"),     # waist_roll
    _action_scale("5020"),     # waist_pitch
    _action_scale("5020"),     # left_shoulder_pitch
    _action_scale("5020"),     # left_shoulder_roll
    _action_scale("5020"),     # left_shoulder_yaw
    _action_scale("5020"),     # left_elbow
    _action_scale("5020"),     # left_wrist_roll
    _action_scale("4010"),     # left_wrist_pitch
    _action_scale("4010"),     # left_wrist_yaw
    _action_scale("5020"),     # right_shoulder_pitch
    _action_scale("5020"),     # right_shoulder_roll
    _action_scale("5020"),     # right_shoulder_yaw
    _action_scale("5020"),     # right_elbow
    _action_scale("5020"),     # right_wrist_roll
    _action_scale("4010"),     # right_wrist_pitch
    _action_scale("4010"),     # right_wrist_yaw
])

# 29-DOF default standing angles in MuJoCo body joint order
_DEFAULT_ANGLES = np.array([
    -0.312,  # left_hip_pitch
     0.0,    # left_hip_roll
     0.0,    # left_hip_yaw
     0.669,  # left_knee
    -0.363,  # left_ankle_pitch
     0.0,    # left_ankle_roll
    -0.312,  # right_hip_pitch
     0.0,    # right_hip_roll
     0.0,    # right_hip_yaw
     0.669,  # right_knee
    -0.363,  # right_ankle_pitch
     0.0,    # right_ankle_roll
     0.0,    # waist_yaw
     0.0,    # waist_roll
     0.0,    # waist_pitch
     0.2,    # left_shoulder_pitch
     0.2,    # left_shoulder_roll
     0.0,    # left_shoulder_yaw
     0.6,    # left_elbow
     0.0,    # left_wrist_roll
     0.0,    # left_wrist_pitch
     0.0,    # left_wrist_yaw
     0.2,    # right_shoulder_pitch
    -0.2,    # right_shoulder_roll
     0.0,    # right_shoulder_yaw
     0.6,    # right_elbow
     0.0,    # right_wrist_roll
     0.0,    # right_wrist_pitch
     0.0,    # right_wrist_yaw
])

# Joint order remapping (from policy_parameters.hpp)
# ISAACLAB_TO_MUJOCO[mujoco_i] = index into policy output for MuJoCo joint i
_ISAACLAB_TO_MUJOCO = np.array([
     0,  3,  6,  9, 13, 17,
     1,  4,  7, 10, 14, 18,
     2,  5,  8, 11, 15, 19, 21, 23, 25, 27,
    12, 16, 20, 22, 24, 26, 28,
])
# MUJOCO_TO_ISAACLAB[mujoco_i] = IsaacLab index of MuJoCo joint i
_MUJOCO_TO_ISAACLAB = np.array([
     0,  6, 12,  1,  7, 13,  2,  8, 14,  3,  9, 15,
    22,  4, 10, 16, 23,  5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
])

# Wrist joint indices (MuJoCo body29 order) used by motion_joint_positions_wrists_10frame_step1
# wrist_joint_isaaclab_order_in_mujoco_index = {19, 26, 20, 27, 21, 28}
_WRIST_IDX_IN_MUJOCO29 = np.array([19, 26, 20, 27, 21, 28])
# Corresponding indices in 43-DOF lerobot state/action:
# mujoco29[0:22] = 43dof[0:22],  mujoco29[22:29] = 43dof[29:36]
_WRIST_IDX_IN_43DOF = np.array([19, 33, 20, 34, 21, 35])

# Encoder input layout (observation_config.yaml encoder_observations order)
# name: (start, end)  — encoder input is 1762 dims total
_ENC_OBS_LAYOUT = {
    "encoder_mode_4":                            (    0,    4),
    "motion_joint_positions_10frame_step5":      (    4,  294),
    "motion_joint_velocities_10frame_step5":     (  294,  584),
    "motion_root_z_position_10frame_step5":      (  584,  594),
    "motion_root_z_position":                    (  594,  595),
    "motion_anchor_orientation":                 (  595,  601),
    "motion_anchor_orientation_10frame_step5":   (  601,  661),
    "motion_joint_positions_lowerbody_10frame_step5":  (  661,  781),
    "motion_joint_velocities_lowerbody_10frame_step5": (  781,  901),
    "vr_3point_local_target":                    (  901,  910),
    "vr_3point_local_orn_target":                (  910,  922),
    "smpl_joints_10frame_step1":                 (  922, 1642),
    "smpl_anchor_orientation_10frame_step1":     ( 1642, 1702),
    "motion_joint_positions_wrists_10frame_step1":    (1702, 1762),
}
_ENC_INPUT_DIM = 1762

# Decoder input layout (observation_config.yaml observations order)
# name: (start, end)  — decoder input is 994 dims total
_DEC_OBS_LAYOUT = {
    "token_state":                           (   0,  64),
    "his_base_angular_velocity_10frame_step1": ( 64,  94),
    "his_body_joint_positions_10frame_step1":  ( 94, 384),
    "his_body_joint_velocities_10frame_step1": (384, 674),
    "his_last_actions_10frame_step1":          (674, 964),
    "his_gravity_dir_10frame_step1":           (964, 994),
}
_DEC_INPUT_DIM = 994
_HISTORY_LEN = 10

# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _quat_mult(q1, q2):
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def _quat_conj(q):
    """Conjugate of quaternion [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def _quat_rotate(q, v):
    """Rotate vector v by quaternion q (all wxyz convention)."""
    q_v = np.array([0.0, v[0], v[1], v[2]])
    return _quat_mult(_quat_mult(q, q_v), _quat_conj(q))[1:]

def _quat_to_rot_matrix(q):
    """Convert unit quaternion [w, x, y, z] to 3×3 rotation matrix."""
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])

def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """SLERP between two unit quaternions [w,x,y,z], t in [0,1]."""
    q1 = q1.copy()
    dot = float(np.dot(q0, q1))
    if dot < 0.0:          # ensure shortest path
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:       # nearly identical — linear + renorm
        return (q0 + t * (q1 - q0)) / np.linalg.norm(q0 + t * (q1 - q0))
    theta0 = np.arccos(dot)
    theta  = theta0 * t
    s0 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta0)
    s1 = np.sin(theta) / np.sin(theta0)
    return s0 * q0 + s1 * q1


def _calc_heading_quat(q_wxyz):
    """Extract the yaw (heading) of quaternion q and return a pure z-axis rotation quaternion.

    Mirrors calc_heading_quat_d in math_utils.hpp:
      1. Rotate [1,0,0] by q → forward direction in world frame.
      2. yaw = atan2(fwd.y, fwd.x)
      3. Return [cos(yaw/2), 0, 0, sin(yaw/2)]  (wxyz, z-axis rotation)
    """
    fwd = _quat_rotate(q_wxyz, np.array([1., 0., 0.]))
    yaw = np.arctan2(fwd[1], fwd[0])
    return np.array([np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)])


def _calc_heading_quat_inv(q_wxyz):
    """Inverse heading quaternion: pure z-axis rotation by -yaw(q).

    Mirrors calc_heading_quat_inv_d in math_utils.hpp.
    """
    fwd = _quat_rotate(q_wxyz, np.array([1., 0., 0.]))
    yaw = np.arctan2(fwd[1], fwd[0])
    return np.array([np.cos(yaw / 2), 0., 0., -np.sin(yaw / 2)])


def _smpl_anchor_ori_6d(base_quat_wxyz, body_root_quat_wxyz):
    """6D rotation for smpl_anchor_orientation_*frame.

    = first 2 columns of R(base_quat^-1 * body_root_quat), extracted row-wise:
      [R[0,0], R[0,1], R[1,0], R[1,1], R[2,0], R[2,1]]
    Mirrors GatherMotionAnchorOrientationMutiFrame in g1_deploy_onnx_ref.cpp.
    NOTE: body_root_quat_wxyz should already have the heading correction applied
    (apply_delta_heading * raw_body_root_quat) before calling this function.
    """
    rel_q = _quat_mult(_quat_conj(base_quat_wxyz), body_root_quat_wxyz)
    R = _quat_to_rot_matrix(rel_q)
    return np.array([R[0, 0], R[0, 1], R[1, 0], R[1, 1], R[2, 0], R[2, 1]])

def _gravity_dir_body(base_quat_wxyz):
    """Gravity direction in body frame = quat_rotate(conj(q), [0, 0, -1])."""
    return _quat_rotate(_quat_conj(base_quat_wxyz), np.array([0.0, 0.0, -1.0]))

def _ang_vel_body(q_prev, q_curr, dt):
    """Approximate body-frame angular velocity from consecutive quaternions [w,x,y,z].

    omega_body ≈ 2 * conj(q_prev) * (q_curr - q_prev) / dt  (imaginary part)
    """
    dq = (q_curr - q_prev) / max(dt, 1e-6)
    omega_quat = _quat_mult(_quat_conj(q_prev), dq)
    return 2.0 * omega_quat[1:]  # imaginary part = [wx, wy, wz]

# ---------------------------------------------------------------------------
# SONIC inference runner
# ---------------------------------------------------------------------------

class SonicRunner:
    """Wraps encoder + decoder ONNX sessions.

    Encoder mode: SMPL (mode_id = 2).
    The encoder expects a FUTURE window of 10 frames (current + next 9), not past.
    The decoder uses a PAST history of 10 frames (oldest first).

    Caller is responsible for providing future-window arrays at each step; see
    SonicRunner.make_windows() for a convenience helper.
    """

    def __init__(self, encoder_path: str, decoder_path: str, fps: float = 20.0,
                 closed_loop: bool = True, inference_backend: Any | None = None):
        if inference_backend is None:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._enc = ort.InferenceSession(encoder_path, providers=providers)
            self._dec = ort.InferenceSession(decoder_path, providers=providers)

            enc_in  = self._enc.get_inputs()[0].shape
            dec_in  = self._dec.get_inputs()[0].shape
            enc_out = self._enc.get_outputs()[0].shape
            dec_out = self._dec.get_outputs()[0].shape
            enc_in_dim = enc_in[1]
            dec_in_dim = dec_in[1]
            enc_out_dim = enc_out[1]
            dec_out_dim = dec_out[1]
            self._inference_backend = None
        else:
            self._enc = inference_backend.encoder_session
            self._dec = inference_backend.decoder_session
            enc_in_dim = inference_backend.encoder_input_dim
            dec_in_dim = inference_backend.decoder_input_dim
            enc_out_dim = inference_backend.token_dim
            dec_out_dim = inference_backend.action_dim
            self._inference_backend = inference_backend
        self._dt = 1.0 / fps

        # Validate model shapes
        assert enc_in_dim == _ENC_INPUT_DIM, f"Encoder expects {_ENC_INPUT_DIM} dims, got {enc_in_dim}"
        assert dec_in_dim == _DEC_INPUT_DIM, f"Decoder expects {_DEC_INPUT_DIM} dims, got {dec_in_dim}"
        assert enc_out_dim == 64,            f"Encoder output must be 64 dims, got {enc_out_dim}"
        assert dec_out_dim == 29,            f"Decoder output must be 29 dims, got {dec_out_dim}"
        print(f"  Encoder: {enc_in} → {enc_out}")
        print(f"  Decoder: {dec_in} → {dec_out}")

        # Decoder history buffers — oldest at index 0, newest at index -1
        self._joint_pos_hist    = deque([np.zeros(29,  dtype=np.float32)] * _HISTORY_LEN, maxlen=_HISTORY_LEN)
        self._joint_vel_hist    = deque([np.zeros(29,  dtype=np.float32)] * _HISTORY_LEN, maxlen=_HISTORY_LEN)
        self._last_action_hist  = deque([np.zeros(29,  dtype=np.float32)] * _HISTORY_LEN, maxlen=_HISTORY_LEN)
        self._gravity_hist      = deque([np.array([0., 0., -1.], dtype=np.float32)] * _HISTORY_LEN, maxlen=_HISTORY_LEN)
        self._ang_vel_hist      = deque([np.zeros(3,   dtype=np.float32)] * _HISTORY_LEN, maxlen=_HISTORY_LEN)
        self._prev_base_quat    = np.array([1., 0., 0., 0.], dtype=np.float64)
        self._prev_joint_pos_il = np.zeros(29, dtype=np.float32)  # IsaacLab order
        self._last_encoder_obs: np.ndarray | None = None
        self._last_decoder_obs: np.ndarray | None = None
        self._last_raw_action: np.ndarray | None = None
        # Closed-loop: previous policy output (IL order, deviation from default_angles).
        # None on the first step, then set to the scaled policy output so that
        # his_body_joint_positions and his_last_actions stay consistent (as in real deployment).
        # Disabled in physics mode (real simulated qpos is fed back instead).
        self._closed_loop = closed_loop
        self._prev_policy_body29_il: np.ndarray | None = None

    def reset_history(self, state43_0: np.ndarray, base_quat_0: np.ndarray,
                      action43_0: np.ndarray | None = None,
                      body29_vel0_mujoco: np.ndarray | None = None,
                      base_ang_vel0: np.ndarray | None = None) -> None:
        """Pre-fill decoder history from the first recorded frame.

        The original teleop had 10 frames of real history before the episode started.
        Initialise all 10 slots with the first-frame state so the decoder receives
        a plausible (static-robot) context instead of all-zeros.
        """
        body29_mujoco = np.concatenate([state43_0[0:22], state43_0[29:36]]).astype(np.float64)
        body29_dev    = body29_mujoco - _DEFAULT_ANGLES
        body29_il = np.zeros(29, dtype=np.float32)
        body29_il[_ISAACLAB_TO_MUJOCO] = body29_dev.astype(np.float32)

        # Initialize his_last_actions.
        # Prefer recorded action (oracle) when provided; otherwise infer from measured state.
        raw_action_il = np.zeros(29, dtype=np.float32)
        if action43_0 is not None:
            body29_act = np.concatenate([action43_0[0:22], action43_0[29:36]]).astype(np.float64)
            act_dev = body29_act - _DEFAULT_ANGLES
            raw_action_il[_ISAACLAB_TO_MUJOCO] = (act_dev / _ACTION_SCALE).astype(np.float32)
        else:
            raw_action_il[_ISAACLAB_TO_MUJOCO] = (body29_dev / _ACTION_SCALE).astype(np.float32)

        gravity0 = _gravity_dir_body(base_quat_0.astype(np.float64)).astype(np.float32)
        if body29_vel0_mujoco is not None:
            body29_vel0_il = np.zeros(29, dtype=np.float32)
            body29_vel0_il[_ISAACLAB_TO_MUJOCO] = body29_vel0_mujoco.astype(np.float32)
        else:
            body29_vel0_il = np.zeros(29, dtype=np.float32)
        ang_vel0 = base_ang_vel0.astype(np.float32) if base_ang_vel0 is not None else np.zeros(3, dtype=np.float32)

        self._joint_pos_hist   = deque([body29_il.copy()]    * _HISTORY_LEN, maxlen=_HISTORY_LEN)
        self._joint_vel_hist   = deque([body29_vel0_il.copy()] * _HISTORY_LEN, maxlen=_HISTORY_LEN)
        self._last_action_hist = deque([raw_action_il.copy()] * _HISTORY_LEN, maxlen=_HISTORY_LEN)
        self._gravity_hist     = deque([gravity0.copy()]     * _HISTORY_LEN, maxlen=_HISTORY_LEN)
        self._ang_vel_hist     = deque([ang_vel0.copy()] * _HISTORY_LEN, maxlen=_HISTORY_LEN)
        self._prev_base_quat    = base_quat_0.astype(np.float64).copy()
        self._prev_joint_pos_il = body29_il.copy()
        self._prev_policy_body29_il = body29_il.copy()

    def sync_decoder_history_from_obs(self, decoder_obs: np.ndarray) -> None:
        """Replace rolling decoder histories from a recorded decoder_obs tensor.

        The decoder layout stores history oldest-first.  This is used at
        recorded-warmup handoff so the first live decoder frame starts from the
        same history window that the C++ deploy recorded.
        """
        dec_obs = decoder_obs.astype(np.float32, copy=False)

        s, e = _DEC_OBS_LAYOUT["his_base_angular_velocity_10frame_step1"]
        self._ang_vel_hist = deque(
            [x.copy() for x in dec_obs[s:e].reshape(_HISTORY_LEN, 3)],
            maxlen=_HISTORY_LEN,
        )

        s, e = _DEC_OBS_LAYOUT["his_body_joint_positions_10frame_step1"]
        self._joint_pos_hist = deque(
            [x.copy() for x in dec_obs[s:e].reshape(_HISTORY_LEN, 29)],
            maxlen=_HISTORY_LEN,
        )

        s, e = _DEC_OBS_LAYOUT["his_body_joint_velocities_10frame_step1"]
        self._joint_vel_hist = deque(
            [x.copy() for x in dec_obs[s:e].reshape(_HISTORY_LEN, 29)],
            maxlen=_HISTORY_LEN,
        )

        s, e = _DEC_OBS_LAYOUT["his_last_actions_10frame_step1"]
        self._last_action_hist = deque(
            [x.copy() for x in dec_obs[s:e].reshape(_HISTORY_LEN, 29)],
            maxlen=_HISTORY_LEN,
        )

        s, e = _DEC_OBS_LAYOUT["his_gravity_dir_10frame_step1"]
        self._gravity_hist = deque(
            [x.copy() for x in dec_obs[s:e].reshape(_HISTORY_LEN, 3)],
            maxlen=_HISTORY_LEN,
        )

        self._prev_joint_pos_il = self._joint_pos_hist[-1].copy()
        self._prev_policy_body29_il = self._joint_pos_hist[-1].copy()

    # ------------------------------------------------------------------
    # Static helpers for offline playback
    # ------------------------------------------------------------------

    @staticmethod
    def precompute(smpl_joints: np.ndarray,
                   body_root_quat: np.ndarray,
                   base_quat: np.ndarray,
                   actions43: np.ndarray):
        """Pre-compute per-frame encoder inputs for an entire episode.

        Applies the same heading correction as the C++ deploy code:
          apply_delta_heading = calc_heading_quat(base_quat[0])
                              * calc_heading_quat_inv(body_root_quat[0])
        This aligns the SMPL reference heading with the robot's initial heading.

        Returns:
            body_root_quat_corr: [T, 4]  heading-corrected body_root_quat per frame
            wrist_all:           [T, 6]  wrist deviations in wrist-IsaacLab order per frame
        """
        # Heading correction (constant for the whole episode)
        init_heading     = _calc_heading_quat(base_quat[0].astype(np.float64))
        data_heading_inv = _calc_heading_quat_inv(body_root_quat[0].astype(np.float64))
        apply_delta_heading = _quat_mult(init_heading, data_heading_inv).astype(np.float32)

        heading_deg = 2 * np.degrees(np.arctan2(apply_delta_heading[3], apply_delta_heading[0]))
        print(f"  Heading correction: {heading_deg:.1f}°  "
              f"(robot yaw: {2*np.degrees(np.arctan2(init_heading[3], init_heading[0])):.1f}°, "
              f"SMPL yaw: {-2*np.degrees(np.arctan2(data_heading_inv[3], data_heading_inv[0])):.1f}°)")

        # Apply heading correction to all body_root_quats
        T = len(body_root_quat)
        body_root_quat_corr = np.array([
            _quat_mult(apply_delta_heading, body_root_quat[t].astype(np.float64))
            for t in range(T)
        ], dtype=np.float32)

        wrist_all = actions43[:, _WRIST_IDX_IN_43DOF].astype(np.float32)
        return body_root_quat_corr, wrist_all

    @staticmethod
    def future_window(arr: np.ndarray, t: int, n: int = _HISTORY_LEN) -> np.ndarray:
        """Return arr[t:t+n] clamped at the last frame: shape [n, ...]."""
        T = len(arr)
        idxs = [min(t + i, T - 1) for i in range(n)]
        return arr[idxs]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self,
             state43:              np.ndarray,   # [43]    measured joint pos, MuJoCo 43-DOF
             action43:             np.ndarray,   # [43]    reference action (hand joints taken from here)
             base_quat:            np.ndarray,   # [4]     wxyz robot base quaternion (IMU)
             smpl_joints_win:      np.ndarray,   # [10,72] FUTURE window: frames [t, t+1, ..., t+9]
             body_root_quat_win:   np.ndarray,   # [10,4]  FUTURE window: heading-corrected body_root_quat
             wrist_win:            np.ndarray,   # [10,6]  FUTURE window: wrist joint deviations
             body29_vel_mujoco:    np.ndarray | None = None,  # [29] measured dq in MuJoCo order
             base_ang_vel:         np.ndarray | None = None,  # [3] measured base angular velocity
             encoder_obs_rec:      np.ndarray | None = None,  # [1762] exact recorded encoder input
             token_rec:            np.ndarray | None = None,  # [64] exact recorded token_state
             decoder_obs_rec:      np.ndarray | None = None,  # [994] exact recorded decoder input
             decoder_history_obs_rec: np.ndarray | None = None,  # [994] recorded history used to seed live decoder
             decoder_action_raw_rec: np.ndarray | None = None,  # [29] exact recorded decoder output
             history_action_raw_rec: np.ndarray | None = None,  # [29] raw action to store for next history step
             q_target_cmd_rec:     np.ndarray | None = None,  # [29] exact recorded q_target command
             ) -> np.ndarray:
        """Run one encoder+decoder step.  Returns joint_targets_43 (43-DOF MuJoCo order).

        The 29 body joints are SONIC policy targets; the 7+7 hand joints are
        copied from action43 unchanged.
        """
        # ---- 1. Prepare per-frame features ----------------------------------
        # Body joints in IsaacLab order (deviation from default_angles).
        # Build body29_il from state43 (always used in physics mode; in kinematic mode
        # replaced by previous policy output after the first step to reduce wobbling).
        body29_mujoco = np.concatenate([state43[0:22], state43[29:36]]).astype(np.float64)
        body29_dev    = body29_mujoco - _DEFAULT_ANGLES
        body29_il_from_state = np.zeros(29, dtype=np.float32)
        body29_il_from_state[_ISAACLAB_TO_MUJOCO] = body29_dev.astype(np.float32)

        if self._closed_loop and self._prev_policy_body29_il is not None:
            body29_il = self._prev_policy_body29_il
        else:
            body29_il = body29_il_from_state

        # Joint velocities in IsaacLab order (prefer recorded dq when available).
        if body29_vel_mujoco is not None:
            body29_vel_il = np.zeros(29, dtype=np.float32)
            body29_vel_il[_ISAACLAB_TO_MUJOCO] = body29_vel_mujoco.astype(np.float32)
        else:
            body29_vel_il = (body29_il - self._prev_joint_pos_il) / self._dt

        # Gravity direction in body frame
        gravity_body = _gravity_dir_body(base_quat).astype(np.float32)

        # Angular velocity in body frame (prefer recorded IMU angular velocity when available).
        if base_ang_vel is not None:
            ang_vel = base_ang_vel.astype(np.float32)
        else:
            ang_vel = _ang_vel_body(self._prev_base_quat, base_quat, self._dt).astype(np.float32)

        # ---- 2. Build/consume encoder obs [1762] -----------------------------
        if encoder_obs_rec is not None:
            enc_obs = encoder_obs_rec.astype(np.float32, copy=False)
        else:
            enc_obs = np.zeros(_ENC_INPUT_DIM, dtype=np.float32)

            # encoder_mode_4: [mode_id, 0, 0, 0] — mode_id=2 for SMPL
            s, e = _ENC_OBS_LAYOUT["encoder_mode_4"]
            enc_obs[s] = 2.0

            # smpl_joints_10frame_step1: 10 future frames × 72
            s, e = _ENC_OBS_LAYOUT["smpl_joints_10frame_step1"]
            enc_obs[s:e] = smpl_joints_win.reshape(-1)

            # smpl_anchor_orientation_10frame_step1: 10 future frames × 6
            # Anchor ori uses current base_quat (not future) + heading-corrected future body_root_quat
            s, e = _ENC_OBS_LAYOUT["smpl_anchor_orientation_10frame_step1"]
            anchor_ori_win = np.array([
                _smpl_anchor_ori_6d(base_quat, body_root_quat_win[fi])
                for fi in range(_HISTORY_LEN)
            ], dtype=np.float32)
            enc_obs[s:e] = anchor_ori_win.reshape(-1)

            # motion_joint_positions_wrists_10frame_step1: 10 future frames × 6
            s, e = _ENC_OBS_LAYOUT["motion_joint_positions_wrists_10frame_step1"]
            enc_obs[s:e] = wrist_win.reshape(-1)

        # ---- 3. Run encoder or consume recorded token [64] -------------------
        if token_rec is not None:
            token = token_rec.astype(np.float32, copy=False)
        else:
            token = self._enc.run(None, {"obs_dict": enc_obs[np.newaxis]})[0][0]  # [64]
        self._last_encoder_obs = enc_obs.copy()

        # ---- 4. Update decoder history buffers (PAST, oldest first) ---------
        if decoder_history_obs_rec is not None:
            self.sync_decoder_history_from_obs(decoder_history_obs_rec)
        else:
            self._ang_vel_hist.append(ang_vel)
            self._joint_pos_hist.append(body29_il)
            self._joint_vel_hist.append(body29_vel_il)
            self._gravity_hist.append(gravity_body)
            # last_action: store the raw policy output from the previous step
            # (populated after inference; first frame uses zeros — already in deque)

        # ---- 5. Build/consume decoder obs [994] ------------------------------
        if decoder_obs_rec is not None:
            dec_obs = decoder_obs_rec.astype(np.float32, copy=False)
        else:
            dec_obs = np.zeros(_DEC_INPUT_DIM, dtype=np.float32)

            s, e = _DEC_OBS_LAYOUT["token_state"]
            dec_obs[s:e] = token

            s, e = _DEC_OBS_LAYOUT["his_base_angular_velocity_10frame_step1"]
            for fi, av_ in enumerate(self._ang_vel_hist):
                dec_obs[s + fi*3 : s + (fi+1)*3] = av_

            s, e = _DEC_OBS_LAYOUT["his_body_joint_positions_10frame_step1"]
            for fi, jp in enumerate(self._joint_pos_hist):
                dec_obs[s + fi*29 : s + (fi+1)*29] = jp

            s, e = _DEC_OBS_LAYOUT["his_body_joint_velocities_10frame_step1"]
            for fi, jv in enumerate(self._joint_vel_hist):
                dec_obs[s + fi*29 : s + (fi+1)*29] = jv

            s, e = _DEC_OBS_LAYOUT["his_last_actions_10frame_step1"]
            for fi, la in enumerate(self._last_action_hist):
                dec_obs[s + fi*29 : s + (fi+1)*29] = la

            s, e = _DEC_OBS_LAYOUT["his_gravity_dir_10frame_step1"]
            for fi, gd in enumerate(self._gravity_hist):
                dec_obs[s + fi*3 : s + (fi+1)*3] = gd

        self._last_decoder_obs = dec_obs.copy()

        # ---- 6. Run decoder or consume recorded output -----------------------
        if decoder_action_raw_rec is not None:
            raw_action = decoder_action_raw_rec.astype(np.float32, copy=False)
        else:
            raw_action = self._dec.run(None, {"obs_dict": dec_obs[np.newaxis]})[0][0]  # [29]
        self._last_raw_action = raw_action.astype(np.float32, copy=True)

        # Store raw action in history for the next step.  The C++ teleop logs
        # last_action before inference, then updates it from TensorRT output
        # after inference.  For replay diagnostics we can therefore roll the
        # recorded TensorRT action while still running Python inference now.
        history_action = raw_action if history_action_raw_rec is None else history_action_raw_rec
        self._last_action_hist.append(history_action.astype(np.float32))

        # ---- 7. Post-process: remap + scale + default_angles ----------------
        if q_target_cmd_rec is not None:
            joint_targets_mujoco29 = q_target_cmd_rec.astype(np.float64, copy=False)
        else:
            # Teleop stores MotorCommand::q_target as float, so mirror that
            # cast when replay computes targets from a freshly inferred action.
            joint_targets_mujoco29 = (
                _DEFAULT_ANGLES + raw_action[_ISAACLAB_TO_MUJOCO] * _ACTION_SCALE
            ).astype(np.float32).astype(np.float64)

        # ---- 8. Update state for next step ----------------------------------
        self._prev_base_quat = base_quat.copy()
        self._prev_joint_pos_il = body29_il.copy()
        # Store scaled policy output as next step's "measured" joint state (IL order, deviation)
        policy_dev_mujoco = (joint_targets_mujoco29 - _DEFAULT_ANGLES).astype(np.float32)
        prev_policy_body29_il = np.zeros(29, dtype=np.float32)
        prev_policy_body29_il[_ISAACLAB_TO_MUJOCO] = policy_dev_mujoco
        self._prev_policy_body29_il = prev_policy_body29_il

        # ---- 9. Assemble 43-DOF output -------------------------------------
        # Body joints from policy; hand joints from recorded action
        out43 = np.array(action43, dtype=np.float64)
        out43[0:22]  = joint_targets_mujoco29[0:22]   # left_leg + right_leg + waist + left_arm
        out43[29:36] = joint_targets_mujoco29[22:29]  # right_arm
        return out43


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_xml(env_name: str) -> mujoco.MjModel:
    rel = _ENV_XML.get(env_name)
    if rel is None:
        raise ValueError(f"Unknown env_name '{env_name}'. Choose from: {list(_ENV_XML)}")
    xml_path = str(_REPO_ROOT / rel)
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
    return mujoco.MjModel.from_xml_path(xml_path)


def _build_joint_indices(model: mujoco.MjModel):
    """Return (body_joint_index, left_hand_index, right_hand_index) — arrays of MuJoCo joint ids."""
    body = []
    for i in range(model.njnt):
        name = model.joint(i).name
        if any(k in name for k in _BODY_JOINT_KEYS):
            body.append(i)

    def _hand_ids(side: str) -> list[int]:
        ids = []
        for suffix in _HAND_JOINT_SUFFIXES:
            name = f"{side}_hand_{suffix}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                ids.append(jid)
        return ids

    left_hand = _hand_ids("left")
    right_hand = _hand_ids("right")
    return np.array(body), np.array(left_hand), np.array(right_hand)


def _build_ctrl_map(model: mujoco.MjModel, joint_ids: np.ndarray) -> np.ndarray:
    """Return ctrl array indices for the given joint IDs (-1 if no actuator found)."""
    ctrl_ids = np.full(len(joint_ids), -1, dtype=int)
    for k in range(model.nu):
        if model.actuator(k).trntype == mujoco.mjtTrn.mjTRN_JOINT:
            jid = model.actuator(k).trnid[0]
            matches = np.where(joint_ids == jid)[0]
            if len(matches):
                ctrl_ids[matches[0]] = k
    return ctrl_ids


def _physics_step(model: mujoco.MjModel, data: mujoco.MjData,
                  body_ctrl_ids: np.ndarray, body29_target: np.ndarray,
                  left_ctrl_ids: np.ndarray, left7_target: np.ndarray,
                  right_ctrl_ids: np.ndarray, right7_target: np.ndarray,
                  n_steps: int = _SIM_STEPS_PER_POLICY,
                  root_jid: int = -1,
                  base_pos_start: np.ndarray | None = None,
                  base_pos_end:   np.ndarray | None = None,
                  base_quat_start: np.ndarray | None = None,
                  base_quat_end:   np.ndarray | None = None):
    """Apply PD torques and advance physics for n_steps sub-steps.

    Mirrors gear_sonic/utils/mujoco_sim/base_sim.py sim_step():
      - FREE_BASE=False → ctrl = torques (no prepended zeros; no root actuators)
      - tau = Kp*(target - q) + Kd*(0 - dq)  (tau_ff=0, dq_des=0)

    If root_jid >= 0 and base_pos/quat are provided, the floating_base_joint is
    driven along the recorded trajectory (kinematically) at each sub-step.
    This replaces the WBC leg controller that kept the robot balanced in the
    original teleop, allowing upper-body joint dynamics to be validated in
    isolation.  Without this, the unactuated legs cause the robot to fall under
    gravity (~90° bq drift within 100 policy steps).
    """
    drive_base = (root_jid >= 0
                  and base_pos_start is not None
                  and base_pos_end   is not None
                  and base_quat_start is not None
                  and base_quat_end   is not None)

    qpos_adr = qvel_adr = 0
    lin_vel = ang_vel_world = np.zeros(3)

    if drive_base:
        qpos_adr = model.jnt_qposadr[root_jid]
        qvel_adr = model.jnt_dofadr[root_jid]
        policy_dt = n_steps * _SIM_DT
        # Constant linear velocity over this policy step
        lin_vel = (base_pos_end - base_pos_start) / policy_dt  # type: ignore[operator]
        # Angular velocity in world frame from quat_start → quat_end
        q_diff = _quat_mult(_quat_conj(base_quat_start), base_quat_end)  # type: ignore[arg-type]
        half_angle = np.arccos(np.clip(abs(float(q_diff[0])), 0.0, 1.0))
        if half_angle < 1e-8:
            ang_vel_world = np.zeros(3)
        else:
            axis_body = q_diff[1:4] / np.sin(half_angle)
            R = _quat_to_rot_matrix(base_quat_start)  # type: ignore[arg-type]
            ang_vel_world = R @ (axis_body * (2.0 * half_angle / policy_dt))

    for k in range(n_steps):
        # Drive free joint along recorded trajectory before each sub-step.
        # This replaces the WBC leg controller: the base follows the recorded
        # path exactly; only upper-body joint dynamics evolve under physics.
        if drive_base:
            frac = k / n_steps
            pos_k  = (1.0 - frac) * base_pos_start + frac * base_pos_end  # type: ignore[operator]
            quat_k = _quat_slerp(base_quat_start, base_quat_end, frac)     # type: ignore[arg-type]
            data.qpos[qpos_adr    :qpos_adr + 3] = pos_k
            data.qpos[qpos_adr + 3:qpos_adr + 7] = quat_k          # wxyz
            data.qvel[qvel_adr    :qvel_adr + 3] = lin_vel
            data.qvel[qvel_adr + 3:qvel_adr + 6] = ang_vel_world

        ctrl = np.zeros(model.nu)

        # Body joint PD
        for j, (cid, kp, kd, tlim) in enumerate(
                zip(body_ctrl_ids, _KP_BODY29, _KD_BODY29, _TORQUE_LIMIT_BODY29)):
            if cid < 0:
                continue
            q  = data.qpos[model.jnt_qposadr[model.actuator(cid).trnid[0]]]
            dq = data.qvel[model.jnt_dofadr[ model.actuator(cid).trnid[0]]]
            tau = kp * (body29_target[j] - q) + kd * (0.0 - dq)
            ctrl[cid] = np.clip(tau, -tlim, tlim)

        # Hand joint PD (left then right)
        for target_arr, ctrl_ids in ((left7_target, left_ctrl_ids),
                                     (right7_target, right_ctrl_ids)):
            for j, cid in enumerate(ctrl_ids):
                if cid < 0:
                    continue
                q  = data.qpos[model.jnt_qposadr[model.actuator(cid).trnid[0]]]
                dq = data.qvel[model.jnt_dofadr[ model.actuator(cid).trnid[0]]]
                tau = _KP_HAND * (target_arr[j] - q) + _KD_HAND * (0.0 - dq)
                ctrl[cid] = np.clip(tau, -_TORQUE_LIMIT_HAND, _TORQUE_LIMIT_HAND)

        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)


def _load_episode(dataset_dir: str, episode: int, sonic: bool = False):
    """Load episode data from Parquet.

    Returns:
        states:         float32 [T, 43] measured joint positions
        actions:        float64 [T, 43] reference actions, or None for state-only playback
        task_indices:   list of task_index per frame
        base_pos:       float64 [T, 3] or None
        base_quat:      float64 [T, 4] wxyz or None
        base_ang_vel:   float64 [T, 3] or None  (decoder history, if available)
        body_dq:        float64 [T, 29] or None (decoder history, if available)
        enc_obs_rec:    float64 [T, 1762] or None
        token_rec:      float64 [T, 64] or None
        dec_obs_rec:    float64 [T, 994] or None
        dec_action_rec: float64 [T, 29] or None
        q_target_cmd:   float64 [T, 29] or None
        smpl_joints:    float32 [T, 72] or None (v2.2+ only, loaded when sonic=True)
        body_root_quat: float32 [T, 4]  or None
    """
    chunk = episode // 1000
    path = os.path.join(
        dataset_dir,
        f"data/chunk-{chunk:03d}/episode_{episode:06d}.parquet",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}")

    schema = pq.read_schema(path)
    action_col = None
    if "action" in schema.names:
        action_col = "action"
    elif "action.wbc" in schema.names:
        action_col = "action.wbc"

    cols = ["observation.state", "task_index"]
    if sonic:
        if action_col is None:
            raise ValueError(
                "SONIC playback requires an action column, but neither 'action' nor "
                "'action.wbc' was found. For pure state playback, omit --sonic_encoder."
            )
        cols.append(action_col)
    has_base_pos = "robot.base_pos" in schema.names
    base_quat_col = None
    if "robot.base_quat" in schema.names:
        base_quat_col = "robot.base_quat"
    elif "observation.root_orientation" in schema.names:
        # Older datasets store the pelvis/root orientation here instead of robot.base_quat.
        base_quat_col = "observation.root_orientation"
    has_base = has_base_pos and base_quat_col == "robot.base_quat"
    if has_base_pos:
        cols.append("robot.base_pos")
    if base_quat_col is not None:
        cols.append(base_quat_col)

    has_smpl = ("pico.smpl_joints" in schema.names and
                "pico.body_root_quat" in schema.names and
                "robot.base_quat" in schema.names)
    has_exact_onnx = all(
        name in schema.names
        for name in (
            "sonic.encoder_obs",
            "sonic.token_state",
            "sonic.decoder_obs",
            "sonic.decoder_action_raw",
            "sonic.q_target_cmd",
        )
    )
    if sonic and not has_smpl and not has_exact_onnx:
        raise ValueError(
            "SONIC mode requires either exact sonic.* ONNX columns or "
            "pico.smpl_joints, pico.body_root_quat, robot.base_quat columns (v2.2+ dataset)."
        )
    if sonic:
        if has_smpl:
            cols += ["pico.smpl_joints", "pico.body_root_quat"]
        if base_quat_col is not None and base_quat_col not in cols:
            cols.append(base_quat_col)
        if "robot.base_ang_vel" in schema.names:
            cols.append("robot.base_ang_vel")
        if "robot.body_dq" in schema.names:
            cols.append("robot.body_dq")
        if "sonic.encoder_obs" in schema.names:
            cols.append("sonic.encoder_obs")
        if "sonic.token_state" in schema.names:
            cols.append("sonic.token_state")
        if "sonic.decoder_obs" in schema.names:
            cols.append("sonic.decoder_obs")
        if "sonic.decoder_action_raw" in schema.names:
            cols.append("sonic.decoder_action_raw")
        if "sonic.q_target_cmd" in schema.names:
            cols.append("sonic.q_target_cmd")

    table = pq.read_table(path, columns=cols)

    states  = np.array([r.as_py() for r in table.column("observation.state")], dtype=np.float32)
    actions = None
    if action_col is not None and action_col in table.column_names:
        actions = np.array([r.as_py() for r in table.column(action_col)], dtype=np.float64)
    task_indices = table.column("task_index").to_pylist()

    base_pos = base_quat = base_ang_vel = body_dq = smpl_joints = body_root_quat = None
    enc_obs_rec = token_rec = dec_obs_rec = dec_action_rec = q_target_cmd = None
    if has_base_pos:
        base_pos = np.array([r.as_py() for r in table.column("robot.base_pos")], dtype=np.float64)
    if base_quat_col is not None:
        base_quat = np.array([r.as_py() for r in table.column(base_quat_col)], dtype=np.float64)
    if sonic:
        if "pico.smpl_joints" in table.column_names:
            smpl_joints = np.array([r.as_py() for r in table.column("pico.smpl_joints")], dtype=np.float32)
        if "pico.body_root_quat" in table.column_names:
            body_root_quat = np.array([r.as_py() for r in table.column("pico.body_root_quat")], dtype=np.float32)
        if base_quat is None and "robot.base_quat" in table.column_names:
            base_quat = np.array([r.as_py() for r in table.column("robot.base_quat")], dtype=np.float64)
        if "robot.base_ang_vel" in table.column_names:
            base_ang_vel = np.array([r.as_py() for r in table.column("robot.base_ang_vel")], dtype=np.float64)
        if "robot.body_dq" in table.column_names:
            body_dq = np.array([r.as_py() for r in table.column("robot.body_dq")], dtype=np.float64)
        if "sonic.encoder_obs" in table.column_names:
            enc_obs_rec = np.array([r.as_py() for r in table.column("sonic.encoder_obs")], dtype=np.float64)
        if "sonic.token_state" in table.column_names:
            token_rec = np.array([r.as_py() for r in table.column("sonic.token_state")], dtype=np.float64)
        if "sonic.decoder_obs" in table.column_names:
            dec_obs_rec = np.array([r.as_py() for r in table.column("sonic.decoder_obs")], dtype=np.float64)
        if "sonic.decoder_action_raw" in table.column_names:
            dec_action_rec = np.array([r.as_py() for r in table.column("sonic.decoder_action_raw")], dtype=np.float64)
        if "sonic.q_target_cmd" in table.column_names:
            q_target_cmd = np.array([r.as_py() for r in table.column("sonic.q_target_cmd")], dtype=np.float64)

    return (states, actions, task_indices,
            base_pos, base_quat, base_ang_vel, body_dq,
            enc_obs_rec, token_rec, dec_obs_rec, dec_action_rec, q_target_cmd,
            smpl_joints, body_root_quat)


def _set_qpos(data: mujoco.MjData, model: mujoco.MjModel,
               body_jids, left_jids, right_jids, state43: np.ndarray,
               root_jid: int | None = None,
               base_pos: np.ndarray | None = None,
               base_quat: np.ndarray | None = None):
    """Set MuJoCo qpos from a 43-DOF joint vector (kinematic, no physics)."""
    body29 = state43[_BODY_IDX]
    left7  = state43[_LEFT_HAND_IDX]
    right7 = state43[_RIGHT_HAND_IDX]
    data.qpos[model.jnt_qposadr[body_jids]] = body29
    if len(left_jids):
        data.qpos[model.jnt_qposadr[left_jids]] = left7
    if len(right_jids):
        data.qpos[model.jnt_qposadr[right_jids]] = right7
    if root_jid is not None:
        adr = model.jnt_qposadr[root_jid]
        if base_pos is not None:
            data.qpos[adr:adr + 3] = base_pos
        if base_quat is not None:
            data.qpos[adr + 3:adr + 7] = base_quat


def _make_video_writer(path: str, width: int, height: int, fps: float):
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            "PyAV is only required when --output_video is used. Install it with "
            "`pip install av`, or omit --output_video for viewer-only playback."
        ) from exc

    container = av.open(path, "w")
    stream = container.add_stream("h264", rate=int(fps))
    stream.width  = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "fast"}
    return container, stream


# ---------------------------------------------------------------------------
# Main playback loop
# ---------------------------------------------------------------------------

def playback(
    dataset_dir: str,
    episode: int,
    env_name: str,
    output_video: str | None,
    cameras: list[str],
    no_viewer: bool,
    fps: float,
    video_width: int,
    video_height: int,
    sonic_runner: "SonicRunner | None",
    compare: bool,
    physics: bool = False,
    upper_body_from_action: bool = False,
    debug_state: bool = False,
):
    # 1. Load data
    use_sonic = sonic_runner is not None
    (states, actions, task_indices,
     base_pos, base_quat, base_ang_vel_rec, body_dq_rec,
     enc_obs_rec, token_rec, dec_obs_rec, dec_action_raw_rec, q_target_cmd_rec,
     smpl_joints, body_root_quat) = _load_episode(dataset_dir, episode, sonic=use_sonic)
    T = len(states)

    task_index = task_indices[0] if task_indices else None
    fixed_base_pos = None
    if base_pos is not None:
        print(f"Root pose loaded: base_pos x range [{base_pos[:,0].min():.3f}, {base_pos[:,0].max():.3f}] m")
    elif base_quat is not None:
        fixed_base_pos = np.tile(np.array([0.0, 0.0, _ROOT_HEIGHT], dtype=np.float64), (T, 1))
        print(
            "Root position not in dataset — using fixed standing height with recorded "
            "pelvis/root orientation."
        )
    else:
        print("Warning: robot.base_pos/base_quat not in dataset — root fixed at (0,0,0.8).")

    # Resolve task description
    task_desc = "(unknown)"
    tasks_file = os.path.join(dataset_dir, "meta/tasks.jsonl")
    if os.path.exists(tasks_file):
        with open(tasks_file) as f:
            for line in f:
                rec = json.loads(line)
                if task_index is not None and rec.get("task_index") == task_index:
                    task_desc = rec.get("task", task_desc)
                    break

    print(f"Episode {episode}: {T} frames @ {fps:.0f} Hz")
    print(f"Task [{task_index}]: {task_desc}")
    if use_sonic:
        mode = "physics (PD+mj_step)" if physics else "kinematic (set_qpos)"
        print(f"SONIC mode: encoder (SMPL mode 2) + decoder inference active  [{mode}]")
        print(f"  Decoder history columns: base_ang_vel={'yes' if base_ang_vel_rec is not None else 'no'}, "
              f"body_dq={'yes' if body_dq_rec is not None else 'no'}")
        print("  Recorded model buffers: "
              f"enc_obs={'yes' if enc_obs_rec is not None else 'no'}, "
              f"token={'yes' if token_rec is not None else 'no'}, "
              f"dec_obs={'yes' if dec_obs_rec is not None else 'no'}, "
              f"dec_action={'yes' if dec_action_raw_rec is not None else 'no'}, "
              f"q_target_cmd={'yes' if q_target_cmd_rec is not None else 'no'}")
        exact_model_io_mode = all(
            x is not None for x in (enc_obs_rec, token_rec, dec_obs_rec, dec_action_raw_rec, q_target_cmd_rec)
        )
        print(f"  Model I/O replay path: {'exact_recorded_buffers' if exact_model_io_mode else 'reconstructed_fallback'}")
        if upper_body_from_action:
            print("  Upper-body override enabled: waist + left/right arm joints are taken from recorded action")

    # 2. Load MuJoCo model
    model = _load_xml(env_name)
    if physics:
        model.opt.timestep = _SIM_DT  # override to match original teleop sim
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    root_jid_list = [i for i in range(model.njnt) if model.joint(i).type == mujoco.mjtJoint.mjJNT_FREE]
    root_jid = root_jid_list[0] if root_jid_list else None
    if root_jid is not None and base_pos is None:
        adr = model.jnt_qposadr[root_jid]
        data.qpos[adr + 2] = _ROOT_HEIGHT
        data.qpos[adr + 3] = 1.0

    body_jids, left_jids, right_jids = _build_joint_indices(model)
    print(f"Joints found — body: {len(body_jids)}, left_hand: {len(left_jids)}, right_hand: {len(right_jids)}")

    # Build ctrl index maps for physics mode
    body_ctrl_ids = left_ctrl_ids = right_ctrl_ids = None
    if physics:
        body_ctrl_ids  = _build_ctrl_map(model, body_jids)
        left_ctrl_ids  = _build_ctrl_map(model, left_jids)
        right_ctrl_ids = _build_ctrl_map(model, right_jids)
        n_body_actd = int((body_ctrl_ids >= 0).sum())
        print(f"Ctrl map — body: {n_body_actd}/{len(body_jids)} actuated, "
              f"left_hand: {int((left_ctrl_ids>=0).sum())}/{len(left_jids)}, "
              f"right_hand: {int((right_ctrl_ids>=0).sum())}/{len(right_jids)}")

    # 3. Resolve requested cameras — fall back to first available on mismatch
    cam_names = [model.cam(i).name for i in range(model.ncam)]
    print(f"Scene cameras: {cam_names}")
    resolved_cams = []
    for cam in cameras:
        if cam in cam_names:
            resolved_cams.append(cam)
        else:
            fallback = cam_names[0] if cam_names else None
            print(f"  Warning: camera '{cam}' not found — falling back to '{fallback}'")
            if fallback and fallback not in resolved_cams:
                resolved_cams.append(fallback)
    if not resolved_cams and cam_names:
        resolved_cams = [cam_names[0]]

    # 4. Set up per-camera offscreen renderers and video writers
    #    If a single output_video path is given and multiple cameras are
    #    requested, each camera gets its own file:
    #      single cam  → output_video as-is
    #      multi cams  → output_video stem + "_<cam>.mp4"
    model.vis.global_.offwidth  = video_width
    model.vis.global_.offheight = video_height

    cam_writers: dict[str, tuple] = {}  # cam → (renderer, container, stream, av module)
    for cam in resolved_cams:
        if output_video is None:
            continue
        if len(resolved_cams) == 1:
            vid_path = output_video
        else:
            p = pathlib.Path(output_video)
            vid_path = str(p.parent / f"{p.stem}_{cam}{p.suffix}")
        renderer = mujoco.Renderer(model, height=video_height, width=video_width)
        container, stream = _make_video_writer(vid_path, video_width, video_height, fps)
        import av
        cam_writers[cam] = (renderer, container, stream, av)
        print(f"Saving video [{cam}] → {vid_path}  ({video_width}×{video_height} @ {fps:.0f} fps)")

    # 5. Launch onscreen viewer
    viewer = None
    if not no_viewer:
        viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
        if viewer is not None:
            try:
                pelvis_id = model.body("pelvis").id
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = pelvis_id
                viewer.cam.distance  = 2.5
                viewer.cam.elevation = -20
                viewer.cam.azimuth   = 135
            except Exception:
                pass

    # 6. Pre-compute encoder future-window arrays (SONIC mode only)
    body_root_quat_corr = wrist_all = None
    if use_sonic:
        if smpl_joints is not None and body_root_quat is not None:
            bq_arr = base_quat if base_quat is not None else np.tile([1., 0., 0., 0.], (T, 1))
            body_root_quat_corr, wrist_all = SonicRunner.precompute(
                smpl_joints, body_root_quat, bq_arr, actions
            )
        else:
            smpl_joints = np.zeros((T, 72), dtype=np.float32)
            body_root_quat_corr = np.tile(
                np.array([1., 0., 0., 0.], dtype=np.float32), (T, 1)
            )
            wrist_all = np.zeros((T, 6), dtype=np.float32)

    # 7. Stats for --compare
    compare_l2s       = [] if compare else None  # SONIC vs recorded actions (reference targets)
    compare_l2s_state = [] if compare else None  # SONIC vs recorded states  (what kinematic shows)
    compare_l2s_bq    = [] if (compare and physics) else None  # sim base_quat vs recorded
    if compare:
        assert compare_l2s is not None and compare_l2s_state is not None

    # 8. Initialize decoder history before replay loop
    if use_sonic:
        bq0 = base_quat[0] if base_quat is not None else np.array([1., 0., 0., 0.], dtype=np.float64)
        # In physics mode also initialize sim state to frame-0 measured joints.
        if physics:
            _set_qpos(data, model, body_jids, left_jids, right_jids, states[0],
                      root_jid=root_jid,
                      base_pos=base_pos[0] if base_pos is not None else None,
                      base_quat=bq0)
            mujoco.mj_forward(model, data)
        # Pre-fill decoder history with frame-0 values.
        # If recorded decoder-history channels exist, use them directly.
        sonic_runner.reset_history(
            states[0],
            bq0,
            actions[0],
            body29_vel0_mujoco=body_dq_rec[0] if body_dq_rec is not None else None,
            base_ang_vel0=base_ang_vel_rec[0] if base_ang_vel_rec is not None else None,
        )
        if physics:
            print(f"  Physics mode initialized from first frame (sim_dt={_SIM_DT}s, "
                  f"{_SIM_STEPS_PER_POLICY} sub-steps/policy-step)")
        else:
            print("  Kinematic mode decoder history initialized from first frame")

    # 9. Replay loop
    dt = 1.0 / fps
    try:
        for i, state in enumerate(states):
            t_start = time.perf_counter()

            bp = base_pos[i] if base_pos is not None else (fixed_base_pos[i] if fixed_base_pos is not None else None)
            bq = base_quat[i] if base_quat is not None else None

            if use_sonic:
                # In physics mode: read current simulated state as the observation;
                # in kinematic mode: use the recorded state (feed-forward).
                if physics:
                    # Read current simulated state (body joints move freely under PD+physics).
                    # The floating_base_joint moves freely, exactly as in the original teleop.
                    sim_body29 = data.qpos[model.jnt_qposadr[body_jids]]
                    sim_left7  = data.qpos[model.jnt_qposadr[left_jids]]  if len(left_jids)  else np.zeros(7)
                    sim_right7 = data.qpos[model.jnt_qposadr[right_jids]] if len(right_jids) else np.zeros(7)
                    sim_state43 = np.zeros(43)
                    sim_state43[0:22]  = sim_body29[0:22]
                    sim_state43[22:29] = sim_left7
                    sim_state43[29:36] = sim_body29[22:29]
                    sim_state43[36:43] = sim_right7
                    # Use recorded base_quat (= original teleop's mj_data.qpos[3:7] at each step).
                    # The simulated free-joint orientation diverges as soon as the robot tips,
                    # which would corrupt smpl_anchor_orientation (encoder) and gravity_dir/ang_vel
                    # (decoder history) — feeding the recorded value keeps all SONIC inputs on
                    # the same distribution as during data collection.
                    bq_sonic = bq if bq is not None else np.array([1., 0., 0., 0.])
                    state_for_sonic = sim_state43.astype(np.float32)
                else:
                    bq_sonic = bq if bq is not None else np.array([1., 0., 0., 0.])
                    state_for_sonic = state

                # Wrist encoder input: use actual future window from the recorded trajectory,
                # matching what the data collection system saw during teleop.
                wrist_win_now = SonicRunner.future_window(wrist_all, i)
                sonic_state43 = sonic_runner.step(
                    state43             = state_for_sonic,
                    action43            = actions[i],
                    base_quat           = bq_sonic,
                    smpl_joints_win     = SonicRunner.future_window(smpl_joints,         i),
                    body_root_quat_win  = SonicRunner.future_window(body_root_quat_corr, i),
                    wrist_win           = wrist_win_now,
                    body29_vel_mujoco   = body_dq_rec[i] if body_dq_rec is not None else None,
                    base_ang_vel        = base_ang_vel_rec[i] if base_ang_vel_rec is not None else None,
                    encoder_obs_rec     = enc_obs_rec[i] if enc_obs_rec is not None else None,
                    token_rec           = token_rec[i] if token_rec is not None else None,
                    decoder_obs_rec     = dec_obs_rec[i] if dec_obs_rec is not None else None,
                    decoder_action_raw_rec = dec_action_raw_rec[i] if dec_action_raw_rec is not None else None,
                    q_target_cmd_rec    = q_target_cmd_rec[i] if q_target_cmd_rec is not None else None,
                )

                # Optional visual/behavioral compatibility mode:
                # Use recorded action for upper-body joints (waist + arms) while
                # keeping SONIC lower-body outputs.
                if upper_body_from_action:
                    sonic_state43[_WAIST_IDX] = actions[i][_WAIST_IDX]
                    sonic_state43[_LEFT_ARM_IDX] = actions[i][_LEFT_ARM_IDX]
                    sonic_state43[_RIGHT_ARM_IDX] = actions[i][_RIGHT_ARM_IDX]

                # Fallback for older datasets that do not contain exact decoder outputs:
                # force last_action history to the recorded action target.
                if dec_action_raw_rec is None:
                    body29_rec = np.concatenate([actions[i][0:22], actions[i][29:36]])
                    rec_dev    = (body29_rec - _DEFAULT_ANGLES).astype(np.float64)
                    raw_rec_il = np.zeros(29, dtype=np.float32)
                    raw_rec_il[_ISAACLAB_TO_MUJOCO] = (rec_dev / _ACTION_SCALE).astype(np.float32)
                    sonic_runner._last_action_hist[-1] = raw_rec_il

                if physics:
                    # Apply PD torques and advance physics.
                    # The free joint is driven along the recorded base trajectory
                    # (replaces the WBC leg controller from the original teleop).
                    body29_target = np.concatenate([sonic_state43[0:22], sonic_state43[29:36]])
                    i_next = min(i + 1, T - 1)
                    bp_next = base_pos[i_next]  if base_pos  is not None else None
                    bq_next = base_quat[i_next] if base_quat is not None else None
                    _physics_step(model, data,
                                  body_ctrl_ids,  body29_target,
                                  left_ctrl_ids,  sonic_state43[22:29],
                                  right_ctrl_ids, sonic_state43[36:43],
                                  root_jid=root_jid if root_jid is not None else -1,
                                  base_pos_start=bp,    base_pos_end=bp_next,
                                  base_quat_start=bq,   base_quat_end=bq_next)
                    display_state = sonic_state43  # for video: show targets
                    # sim_body29 is always set above before reaching this branch
                    _sb29: np.ndarray = sim_body29  # type: ignore[possibly-unbound]
                    if compare:
                        # A: sim qpos (BEFORE this step's control) vs recorded observation.state
                        compare_l2s_state.append(float(np.linalg.norm(_sb29 - state[_BODY_IDX])))  # type: ignore[union-attr]
                        # B: SONIC output vs recorded action (= original policy output)
                        sonic_body29 = np.concatenate([sonic_state43[0:22], sonic_state43[29:36]])
                        compare_l2s.append(float(np.linalg.norm(sonic_body29 - actions[i][_BODY_IDX])))  # type: ignore[union-attr]
                        # C: simulated base_quat vs recorded — tracks free-joint drift (degrees)
                        if root_jid is not None and compare_l2s_bq is not None:
                            adr = model.jnt_qposadr[root_jid]
                            bq_sim = data.qpos[adr + 3:adr + 7]
                            dot = float(np.clip(abs(np.dot(bq_sim, bq_sonic)), 0.0, 1.0))
                            compare_l2s_bq.append(2.0 * np.degrees(np.arccos(dot)))
                else:
                    display_state = sonic_state43
                    if compare:
                        assert compare_l2s is not None and compare_l2s_state is not None
                        sonic_body29 = np.concatenate([sonic_state43[0:22], sonic_state43[29:36]])
                        ref_body29   = actions[i][_BODY_IDX]
                        obs_body29   = state[_BODY_IDX]
                        compare_l2s.append(float(np.linalg.norm(sonic_body29 - ref_body29)))
                        compare_l2s_state.append(float(np.linalg.norm(sonic_body29 - obs_body29)))
            else:
                display_state = state

            if not physics:
                _set_qpos(data, model, body_jids, left_jids, right_jids, display_state,
                          root_jid=root_jid, base_pos=bp, base_quat=bq)
                if debug_state and not use_sonic and (i < 10 or (i + 1) % 100 == 0 or i == T - 1):
                    left_leg = display_state[0:6]
                    right_leg = display_state[6:12]
                    print(
                        "    pure_state "
                        f"base_pos={np.round(bp, 4) if bp is not None else None} "
                        f"left_leg={np.round(left_leg, 4)} "
                        f"right_leg={np.round(right_leg, 4)}"
                    )
            mujoco.mj_forward(model, data)

            # Render each camera
            for cam, (renderer, container, stream, av) in cam_writers.items():
                renderer.update_scene(data, camera=cam)
                rgb = renderer.render()
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                frame.pts = i
                for pkt in stream.encode(frame):
                    container.mux(pkt)

            if viewer is not None and viewer.is_running():
                viewer.sync()

            elapsed = time.perf_counter() - t_start
            sleep_s = dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

            if i < 10 or (i + 1) % 100 == 0 or i == T - 1:
                msg = f"  Frame {i+1}/{T}"
                if compare and compare_l2s and compare_l2s_state:
                    msg += f"  L2_action={compare_l2s[-1]:.4f}  L2_state={compare_l2s_state[-1]:.4f}"
                    if compare_l2s_bq:
                        msg += f"  L2_bq={compare_l2s_bq[-1]:.2f}°"
                print(msg)

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        if viewer is not None:
            viewer.close()
        for cam, (renderer, container, stream) in cam_writers.items():
            for pkt in stream.encode():
                container.mux(pkt)
            container.close()

    if compare and compare_l2s:
        arr  = np.array(compare_l2s)
        arrs = np.array(compare_l2s_state)
        print(f"\nL2 vs reference actions (SONIC targets vs teleop targets):")
        print(f"  mean={arr.mean():.4f}  median={np.median(arr):.4f}  max={arr.max():.4f}")
        if physics:
            print(f"L2 simulated qpos vs recorded observation.state (should be ~0 if replay is exact):")
        else:
            print(f"L2 vs observed states  (SONIC targets vs kinematic playback):")
        print(f"  mean={arrs.mean():.4f}  median={np.median(arrs):.4f}  max={arrs.max():.4f}")
        if compare_l2s_bq:
            arrb = np.array(compare_l2s_bq)
            print(f"L2 sim base_quat vs recorded (free-joint drift, degrees):")
            print(f"  mean={arrb.mean():.2f}°  median={np.median(arrb):.2f}°  max={arrb.max():.2f}°")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Replay a LeRobot episode in MuJoCo.")
    parser.add_argument("--dataset_dir", required=True, help="Path to LeRobot dataset root.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--env_name",
        default="kitchen_pnp_apple",
        choices=list(_ENV_XML),
        help="MuJoCo scene (default: kitchen_pnp_apple).",
    )
    parser.add_argument("--output_video", default=None,
                        help="Output MP4 path. With multiple --cameras, camera name is appended as suffix.")
    # Camera selection (single legacy flag kept for compat, or multi via --cameras)
    parser.add_argument("--camera",  default=None,
                        help="Single camera name (legacy shorthand; use --cameras for multiple).")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="One or more camera names for video output (e.g. overview head_camera_left).")
    parser.add_argument("--no_viewer", action="store_true")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--video_width",  type=int, default=640)
    parser.add_argument("--video_height", type=int, default=360)
    # SONIC inference flags
    parser.add_argument("--sonic_encoder", default=None,
                        help="Path to model_encoder.onnx (enables SONIC inference mode).")
    parser.add_argument("--sonic_decoder", default=None,
                        help="Path to model_decoder.onnx (required with --sonic_encoder).")
    parser.add_argument("--compare", action="store_true",
                        help="Print per-frame L2 between SONIC predicted and recorded body joints.")
    parser.add_argument("--physics", action="store_true",
                        help="Physics replay: apply SONIC targets via PD+mj_step instead of "
                             "kinematic set_qpos. Replicates the original teleop simulation "
                             "exactly; --compare L2_state should be ~0 if replay is correct.")
    parser.add_argument("--upper_body_from_action", action="store_true",
                        help="Override SONIC upper-body joints with recorded action targets "
                             "(waist + left/right 7-DoF arms) to improve visual match.")
    parser.add_argument("--debug_state", action="store_true",
                        help="Print recorded base/leg state values as they are written to qpos.")
    args = parser.parse_args()

    # Resolve camera list
    if args.cameras:
        cameras = args.cameras
    elif args.camera:
        cameras = [args.camera]
    else:
        cameras = ["overview"]

    # Build SONIC runner if requested
    sonic_runner = None
    if args.sonic_encoder:
        if not args.sonic_decoder:
            parser.error("--sonic_decoder is required when --sonic_encoder is given")
        print("Loading SONIC models...")
        # Always closed_loop=False: use the actual observed state (sim or recorded) as the
        # joint-position history, matching the original teleop which used body_q_measured.
        sonic_runner = SonicRunner(args.sonic_encoder, args.sonic_decoder, fps=args.fps,
                                   closed_loop=False)

    playback(
        dataset_dir  = args.dataset_dir,
        episode      = args.episode,
        env_name     = args.env_name,
        output_video = args.output_video,
        cameras      = cameras,
        no_viewer    = args.no_viewer,
        fps          = args.fps,
        video_width  = args.video_width,
        video_height = args.video_height,
        sonic_runner = sonic_runner,
        compare      = args.compare,
        physics      = args.physics,
        upper_body_from_action = args.upper_body_from_action,
        debug_state  = args.debug_state,
    )


if __name__ == "__main__":
    main()
