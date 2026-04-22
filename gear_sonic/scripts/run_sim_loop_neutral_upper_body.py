"""Run MuJoCo sim loop with a neutral upper-body default posture.

This is a variant of `run_sim_loop.py` for balance testing:
- arms are set to a straight, unbent neutral posture
- shoulder side lean is removed (left/right shoulder roll set to 0)

The key override is `loco_upper_body_dof_pos`, which is normally:
  [waist(3), left shoulder+elbow(4), left wrist(3), right shoulder+elbow(4), right wrist(3)]
and in the default config contains non-zero shoulder roll / elbow bend.
"""

from dataclasses import dataclass
from typing import Dict, Literal

import mujoco
import tyro

from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.data.robot_model.robot_model import RobotModel

@dataclass
class NeutralUpperBodyConfig(SimLoopConfig):
    upper_body_pose_preset: Literal["natural_relaxed", "neutral_zero"] = "natural_relaxed"
    """Initial upper-body pose preset loaded at startup only (no hard lock)."""


ArgsConfig = NeutralUpperBodyConfig

# 17 values order:
# waist(3) + left shoulder/elbow(4) + left wrist(3) +
# right shoulder/elbow(4) + right wrist(3)
_UPPER_BODY_PRESETS = {
    # All zeros; may still look slightly bent due to robot link geometry.
    "neutral_zero": [0.0] * 17,
    # Natural relaxed posture: no side lean, near-straight elbows.
    "natural_relaxed": [
        0.0, 0.0, 0.0,      # waist
        0.0, 0.0, 0.0, 1.2,  # left shoulder/elbow (near-straight for this MJCF)
        0.0, 0.0, 0.0,      # left wrist
        0.0, 0.0, 0.0, 1.2,  # right shoulder/elbow (near-straight for this MJCF)
        0.0, 0.0, 0.0,      # right wrist
    ],
}

# Direct MuJoCo joint targets for "natural, no-bend" startup pose.
# This is applied once at startup and on each environment reset only.
_NATURAL_ARM_JOINT_TARGETS = {
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 1.2,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 1.2,
}

# 29-DoF indices in config arrays (MuJoCo body29 order):
# left: shoulder_pitch(15), shoulder_roll(16), shoulder_yaw(17), elbow(18)
# right: shoulder_pitch(22), shoulder_roll(23), shoulder_yaw(24), elbow(25)
_NATURAL_ARM_IDX_TO_VAL = {
    15: 0.0,
    16: 0.0,
    17: 0.0,
    18: 1.2,
    22: 0.0,
    23: 0.0,
    24: 0.0,
    25: 1.2,
}


def _apply_initial_arm_pose(simulator, joint_targets: Dict[str, float]) -> None:
    """Apply upper-body joint pose once (or on reset), without hard locking."""
    sim_env = simulator.sim_env
    model = sim_env.mj_model
    data = sim_env.mj_data
    applied = {}

    for jname, target in joint_targets.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        qaddr = model.jnt_qposadr[jid]
        data.qpos[qaddr] = float(target)
        applied[jname] = float(data.qpos[qaddr])

    mujoco.mj_forward(model, data)
    if applied:
        print(f"[NeutralUpperBody] Applied startup arm pose: {applied}")
    else:
        print("[NeutralUpperBody] Warning: no arm joints were applied (joint names not found).")


def _install_reset_pose_hook(simulator, joint_targets: Dict[str, float]) -> None:
    """Re-apply startup pose after each reset (still not a runtime hard lock)."""
    sim_env = simulator.sim_env
    original_reset = sim_env.reset

    def _reset_with_pose():
        original_reset()
        _apply_initial_arm_pose(simulator, joint_targets)

    sim_env.reset = _reset_with_pose


def _override_default_arm_angles(config: Dict[str, any]) -> None:
    """Override startup default arrays used by controller/command init paths."""
    for key in ("DEFAULT_DOF_ANGLES", "DEFAULT_MOTOR_ANGLES"):
        arr = config.get(key, None)
        if not isinstance(arr, list) or len(arr) < 29:
            continue
        for idx, val in _NATURAL_ARM_IDX_TO_VAL.items():
            arr[idx] = float(val)
    print("[NeutralUpperBody] Overrode DEFAULT_DOF_ANGLES/DEFAULT_MOTOR_ANGLES arm entries.")


class SimWrapper:
    def __init__(self, robot_model: RobotModel, env_name: str, config: Dict[str, any], **kwargs):
        self.robot_model = robot_model
        self.config = config

        init_channel(config=self.config)

        self.sim = SimulatorFactory.create_simulator(
            config=self.config,
            env_name=env_name,
            **kwargs,
        )


def main(config: ArgsConfig):
    wbc_config = config.load_wbc_yaml()
    wbc_config["ENV_NAME"] = config.env_name
    preset = _UPPER_BODY_PRESETS[config.upper_body_pose_preset]
    _override_default_arm_angles(wbc_config)

    # Initial-load only preset (no hard lock): these values are used by startup/stand logic.
    wbc_config["loco_upper_body_dof_pos"] = preset.copy()
    # Also neutralize any configured start pose used by mimic/stand pipelines.
    if "start_upper_body_dof_pos" in wbc_config and isinstance(wbc_config["start_upper_body_dof_pos"], dict):
        for k in list(wbc_config["start_upper_body_dof_pos"].keys()):
            wbc_config["start_upper_body_dof_pos"][k] = preset.copy()
    print(f"[NeutralUpperBody] upper_body_pose_preset={config.upper_body_pose_preset}")
    print("[NeutralUpperBody] Applied at load/reset only (no hard lock).")

    if config.enable_image_publish:
        assert config.enable_offscreen, "enable_offscreen must be True when enable_image_publish is True"

    robot_model = instantiate_g1_robot_model()

    head_cam_kwargs = {}
    if config.head_cam:
        head_cam_kwargs = {
            "camera_configs": {
                "head_camera_left": {"height": 480, "width": 640},
                "head_camera_right": {"height": 480, "width": 640},
            },
            "head_cam_shm_name": "pico_head_cam",
        }

    if config.enable_image_publish and not config.head_cam:
        head_cam_kwargs["camera_configs"] = {"head_camera": {"height": 480, "width": 640}}

    sim_wrapper = SimWrapper(
        robot_model=robot_model,
        env_name=config.env_name,
        config=wbc_config,
        onscreen=wbc_config.get("ENABLE_ONSCREEN", True),
        offscreen=wbc_config.get("ENABLE_OFFSCREEN", False) or config.head_cam or config.enable_image_publish,
        enable_image_publish=config.enable_image_publish,
        base_state_port=config.base_state_port,
        **head_cam_kwargs,
    )

    # Ensure the *actual loaded MuJoCo joint state* starts from natural no-bend arms.
    _install_reset_pose_hook(sim_wrapper.sim, _NATURAL_ARM_JOINT_TARGETS)
    _apply_initial_arm_pose(sim_wrapper.sim, _NATURAL_ARM_JOINT_TARGETS)

    SimulatorFactory.start_simulator(
        sim_wrapper.sim,
        as_thread=False,
        enable_image_publish=config.enable_image_publish,
        mp_start_method=config.mp_start_method,
        camera_port=config.camera_port,
    )


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
