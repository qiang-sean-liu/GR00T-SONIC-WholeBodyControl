"""
Sonic VLA data exporter with extra low-level command/state debug columns.

This is a wrapper around ``run_data_exporter.py``. It keeps the normal dataset
fields and additionally records the low-level command fields needed to replay
Action-PD more exactly:

  - body motor_cmd q/dq/kp/kd/tau
  - left hand motor_cmd q/dq/kp/kd/tau
  - right hand motor_cmd q/dq/kp/kd/tau
  - MuJoCo qpos/qvel snapshots and timestamps, when published

Run from repo root:
    python gear_sonic/scripts/run_data_exporter_lowcmd_debug.py --task-prompt "pick up the cup"

Note: this script can only save fields that are present in the incoming ZMQ
messages. Missing command/state fields are saved as NaN arrays and reported once.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import tyro

from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.features_sonic_vla import (
    get_features_sonic_vla,
    get_g1_robot_model,
    get_modality_config_sonic_vla,
    get_wrist_camera_features,
    get_wrist_camera_modality_config,
)
from gear_sonic.scripts.run_data_exporter import (
    GrootDataCollector,
    SonicDataExporterConfig,
    TextToSpeech,
    poll_robot_config_zmq,
)

_BODY_MOTORS = 29
_HAND_MOTORS = 7
_G1_43DOF_QPOS = 50  # free root (7) + 43 actuated joints
_G1_43DOF_QVEL = 49  # free root (6) + 43 actuated joints
_HAND_KP_DEFAULT = 1.5
_HAND_KD_DEFAULT = 0.1


def _names(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_{i}" for i in range(n)]


def _add_array_feature(features: dict, key: str, length: int) -> None:
    features[key] = {
        "dtype": "float64",
        "shape": (length,),
        "names": _names(key.replace(".", "_"), length),
    }


def _add_scalar_feature(features: dict, key: str) -> None:
    features[key] = {
        "dtype": "float64",
        "shape": (1,),
        "names": [key.replace(".", "_")],
    }


def _add_lowcmd_debug_features(features: dict) -> None:
    for prefix, length in (
        ("robot.motor", _BODY_MOTORS),
        ("robot.left_hand_motor", _HAND_MOTORS),
        ("robot.right_hand_motor", _HAND_MOTORS),
    ):
        for field in ("q", "dq", "kp", "kd", "tau"):
            _add_array_feature(features, f"{prefix}_{field}", length)

    _add_array_feature(features, "robot.mujoco_qpos", _G1_43DOF_QPOS)
    _add_array_feature(features, "robot.mujoco_qvel", _G1_43DOF_QVEL)
    _add_scalar_feature(features, "robot.control_timestamp")
    _add_scalar_feature(features, "robot.sim_timestamp")


class LowCmdDebugDataCollector(GrootDataCollector):
    """Collector that appends low-level command and MuJoCo state debug fields."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._missing_debug_field_warnings: set[str] = set()

    def _warn_missing_once(self, source_key: str, output_key: str) -> None:
        if output_key in self._missing_debug_field_warnings:
            return
        self._missing_debug_field_warnings.add(output_key)
        self._print_and_say(
            f"[LowCmdDebug] Missing source field {source_key!r}; "
            f"saving {output_key!r} as NaN until publisher provides it.",
            say=False,
        )

    def _array_from_sources(
        self,
        source: dict,
        source_keys: tuple[str, ...],
        output_key: str,
        length: int,
        default: float | np.ndarray | None = None,
    ) -> np.ndarray:
        for source_key in source_keys:
            if source_key in source:
                arr = np.asarray(source[source_key], dtype=np.float64).reshape(-1)
                out = np.full(length, np.nan, dtype=np.float64)
                n = min(length, arr.size)
                out[:n] = arr[:n]
                return out

        if default is not None:
            if isinstance(default, np.ndarray):
                arr = np.asarray(default, dtype=np.float64).reshape(-1)
                out = np.full(length, np.nan, dtype=np.float64)
                n = min(length, arr.size)
                out[:n] = arr[:n]
                return out
            return np.full(length, float(default), dtype=np.float64)

        self._warn_missing_once(" or ".join(source_keys), output_key)
        return np.full(length, np.nan, dtype=np.float64)

    def _scalar_from_sources(
        self,
        source: dict,
        source_keys: tuple[str, ...],
        output_key: str,
    ) -> np.ndarray:
        for source_key in source_keys:
            if source_key in source:
                arr = np.asarray(source[source_key], dtype=np.float64).reshape(-1)
                if arr.size:
                    return np.array([float(arr[0])], dtype=np.float64)

        self._warn_missing_once(" or ".join(source_keys), output_key)
        return np.array([np.nan], dtype=np.float64)

    def _add_cpp_state_features(self, frame_data: dict, proprio: dict) -> None:
        super()._add_cpp_state_features(frame_data, proprio)
        self._add_lowcmd_debug_frame_features(frame_data, proprio)

    def _add_lowcmd_debug_frame_features(self, frame_data: dict, proprio: dict) -> None:
        """Add exact command/state columns when available in ZMQ messages."""
        sim_source = (
            {**proprio, **self.latest_base_state_msg}
            if self.latest_base_state_msg is not None
            else proprio
        )

        mappings: tuple[tuple[str, tuple[str, ...], int], ...] = (
            ("robot.motor_q", ("motor_q", "body_motor_q", "body_motor_cmd_q", "q_target_cmd", "last_action"), _BODY_MOTORS),
            ("robot.motor_dq", ("motor_dq", "body_motor_dq", "body_motor_cmd_dq", "dq_target_cmd"), _BODY_MOTORS),
            ("robot.motor_kp", ("motor_kp", "body_motor_kp", "body_motor_cmd_kp"), _BODY_MOTORS),
            ("robot.motor_kd", ("motor_kd", "body_motor_kd", "body_motor_cmd_kd"), _BODY_MOTORS),
            ("robot.motor_tau", ("motor_tau", "body_motor_tau", "body_motor_cmd_tau", "tau_ff"), _BODY_MOTORS),
            ("robot.left_hand_motor_q", ("left_hand_motor_q", "left_hand_cmd_q", "left_dex3_cmd_q"), _HAND_MOTORS),
            ("robot.left_hand_motor_dq", ("left_hand_motor_dq", "left_hand_cmd_dq", "left_dex3_cmd_dq"), _HAND_MOTORS),
            ("robot.left_hand_motor_kp", ("left_hand_motor_kp", "left_hand_cmd_kp", "left_dex3_cmd_kp"), _HAND_MOTORS),
            ("robot.left_hand_motor_kd", ("left_hand_motor_kd", "left_hand_cmd_kd", "left_dex3_cmd_kd"), _HAND_MOTORS),
            ("robot.left_hand_motor_tau", ("left_hand_motor_tau", "left_hand_cmd_tau", "left_dex3_cmd_tau"), _HAND_MOTORS),
            ("robot.right_hand_motor_q", ("right_hand_motor_q", "right_hand_cmd_q", "right_dex3_cmd_q"), _HAND_MOTORS),
            ("robot.right_hand_motor_dq", ("right_hand_motor_dq", "right_hand_cmd_dq", "right_dex3_cmd_dq"), _HAND_MOTORS),
            ("robot.right_hand_motor_kp", ("right_hand_motor_kp", "right_hand_cmd_kp", "right_dex3_cmd_kp"), _HAND_MOTORS),
            ("robot.right_hand_motor_kd", ("right_hand_motor_kd", "right_hand_cmd_kd", "right_dex3_cmd_kd"), _HAND_MOTORS),
            ("robot.right_hand_motor_tau", ("right_hand_motor_tau", "right_hand_cmd_tau", "right_dex3_cmd_tau"), _HAND_MOTORS),
        )

        for output_key, source_keys, length in mappings:
            default = None
            if output_key in {"robot.motor_dq", "robot.motor_tau"}:
                default = 0.0
            elif output_key.endswith("_motor_dq") or output_key.endswith("_motor_tau"):
                default = 0.0
            elif output_key.endswith("_motor_kp") and "hand" in output_key:
                default = _HAND_KP_DEFAULT
            elif output_key.endswith("_motor_kd") and "hand" in output_key:
                default = _HAND_KD_DEFAULT
            frame_data[output_key] = self._array_from_sources(
                proprio, source_keys, output_key, length, default=default
            )

        frame_data["robot.mujoco_qpos"] = self._array_from_sources(
            sim_source,
            ("qpos", "mujoco_qpos", "sim_qpos"),
            "robot.mujoco_qpos",
            _G1_43DOF_QPOS,
        )
        frame_data["robot.mujoco_qvel"] = self._array_from_sources(
            sim_source,
            ("qvel", "mujoco_qvel", "sim_qvel"),
            "robot.mujoco_qvel",
            _G1_43DOF_QVEL,
        )
        frame_data["robot.control_timestamp"] = self._scalar_from_sources(
            proprio,
            ("control_timestamp", "ros_timestamp", "timestamp"),
            "robot.control_timestamp",
        )
        frame_data["robot.sim_timestamp"] = self._scalar_from_sources(
            sim_source,
            ("sim_timestamp", "mujoco_timestamp", "time", "timestamp", "ros_timestamp"),
            "robot.sim_timestamp",
        )


def main(config: SonicDataExporterConfig):
    g1_rm = get_g1_robot_model()

    dataset_features = get_features_sonic_vla(g1_rm)
    dataset_features["robot.base_pos"] = {
        "dtype": "float64",
        "shape": (3,),
        "names": ["base_x", "base_y", "base_z"],
    }
    dataset_features["robot.base_quat"] = {
        "dtype": "float64",
        "shape": (4,),
        "names": ["base_qw", "base_qx", "base_qy", "base_qz"],
    }
    _add_lowcmd_debug_features(dataset_features)

    modality_config = get_modality_config_sonic_vla(g1_rm)
    modality_config.setdefault("state", {})["base_pos"] = {
        "start": 0,
        "end": 3,
        "original_key": "robot.base_pos",
    }
    modality_config.setdefault("state", {})["base_quat"] = {
        "start": 0,
        "end": 4,
        "original_key": "robot.base_quat",
        "rotation_type": "quaternion",
    }

    if config.record_wrist_cameras:
        print("[Camera] Wrist cameras enabled - adding to dataset schema")
        dataset_features.update(get_wrist_camera_features())
        wrist_modality = get_wrist_camera_modality_config()
        for key, value in wrist_modality.items():
            if key in modality_config:
                modality_config[key].update(value)
            else:
                modality_config[key] = value

    text_to_speech = TextToSpeech() if config.text_to_speech else None

    robot_config = poll_robot_config_zmq(
        config.state_zmq_host, config.state_zmq_port, config.robot_config_timeout
    )

    data_exporter = Gr00tDataExporter.create(
        save_root=f"{config.root_output_dir}/{config.dataset_name}",
        fps=config.data_collection_frequency,
        features=dataset_features,
        modality_config=modality_config,
        task=config.task_prompt,
        script_config={
            **robot_config,
            **asdict(config),
            "records_base_pos": True,
            "records_lowcmd_debug": True,
            "lowcmd_debug_note": (
                "NaN means the corresponding ZMQ publisher field was not present."
            ),
        },
    )

    data_collector = LowCmdDebugDataCollector(
        frequency=config.data_collection_frequency,
        data_exporter=data_exporter,
        robot_model=g1_rm,
        camera_host=config.camera_host,
        camera_port=config.camera_port,
        text_to_speech=text_to_speech,
        sonic_data_zmq_host=config.sonic_zmq_host,
        sonic_data_zmq_port=config.sonic_zmq_port,
        state_zmq_host=config.state_zmq_host,
        state_zmq_port=config.state_zmq_port,
        base_state_zmq_host=config.base_state_zmq_host,
        base_state_zmq_port=config.base_state_zmq_port,
    )
    data_collector.run()


if __name__ == "__main__":
    cfg = tyro.cli(SonicDataExporterConfig)
    if cfg.dataset_name is None:
        cfg.dataset_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-lowcmd-debug")
    main(cfg)
