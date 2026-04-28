"""Variant of run_data_exporter.py that also records robot base pose.

Adds these dataset features:
  - robot.base_pos
  - robot.base_quat

The script reuses the existing Sonic exporter pipeline and only extends the
schema plus the per-frame state extraction. It is meant to be used instead of
run_data_exporter.py when you want playback-ready root pose fields saved into
the LeRobot dataset.
"""

from __future__ import annotations

from datetime import datetime

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


_DEFAULT_BASE_POS = np.array([0.0, 0.0, 0.8], dtype=np.float64)
_DEFAULT_BASE_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
_BASE_POS_SOURCE = None
_BASE_QUAT_SOURCE = None


def _set_source_once(kind: str, source: str) -> None:
    global _BASE_POS_SOURCE, _BASE_QUAT_SOURCE
    if kind == "pos" and _BASE_POS_SOURCE is None:
        _BASE_POS_SOURCE = source
        print(f"[BasePose] base_pos source: {source}")
    elif kind == "quat" and _BASE_QUAT_SOURCE is None:
        _BASE_QUAT_SOURCE = source
        print(f"[BasePose] base_quat source: {source}")


def _get_features_with_base_pose(robot_model) -> dict:
    features = get_features_sonic_vla(robot_model)
    features.update(
        {
            "robot.base_pos": {
                "dtype": "float64",
                "shape": (3,),
                "names": ["base_x", "base_y", "base_z"],
            },
            "robot.base_quat": {
                "dtype": "float64",
                "shape": (4,),
                "names": ["base_qw", "base_qx", "base_qy", "base_qz"],
            },
        }
    )
    return features


def _get_modality_config_with_base_pose(robot_model) -> dict:
    modality = get_modality_config_sonic_vla(robot_model)
    modality.setdefault("state", {}).update(
        {
            "base_pos": {
                "start": 0,
                "end": 3,
                "original_key": "robot.base_pos",
            },
            "base_quat": {
                "start": 0,
                "end": 4,
                "original_key": "robot.base_quat",
                "rotation_type": "quaternion",
            },
        }
    )
    return modality


def _extract_base_pos(proprio: dict) -> np.ndarray:
    if "base_pos" in proprio:
        arr = np.asarray(proprio["base_pos"], dtype=np.float64).reshape(-1)
        if arr.size >= 3:
            _set_source_once("pos", "base_pos")
            return arr[:3]

    if "base_trans_target" in proprio:
        arr = np.asarray(proprio["base_trans_target"], dtype=np.float64).reshape(-1)
        if arr.size >= 3:
            _set_source_once("pos", "base_trans_target")
            return arr[:3]

    if "base_trans_measured" in proprio:
        arr = np.asarray(proprio["base_trans_measured"], dtype=np.float64).reshape(-1)
        if arr.size >= 3:
            _set_source_once("pos", "base_trans_measured")
            return arr[:3]

    if "floating_base_pose" in proprio:
        arr = np.asarray(proprio["floating_base_pose"], dtype=np.float64).reshape(-1)
        if arr.size >= 3:
            _set_source_once("pos", "floating_base_pose[:3]")
            return arr[:3]

    if "base_pose" in proprio:
        arr = np.asarray(proprio["base_pose"], dtype=np.float64).reshape(-1)
        if arr.size >= 3:
            _set_source_once("pos", "base_pose[:3]")
            return arr[:3]

    odo_state = proprio.get("odo_state")
    if isinstance(odo_state, dict) and "position" in odo_state:
        arr = np.asarray(odo_state["position"], dtype=np.float64).reshape(-1)
        if arr.size >= 3:
            _set_source_once("pos", "odo_state.position")
            return arr[:3]

    _set_source_once("pos", "default_[0,0,0.8]")
    return _DEFAULT_BASE_POS.copy()


def _extract_base_quat(proprio: dict) -> np.ndarray:
    if "base_quat" in proprio:
        arr = np.asarray(proprio["base_quat"], dtype=np.float64).reshape(-1)
        if arr.size >= 4:
            _set_source_once("quat", "base_quat")
            return arr[:4]

    if "base_quat_measured" in proprio:
        arr = np.asarray(proprio["base_quat_measured"], dtype=np.float64).reshape(-1)
        if arr.size >= 4:
            _set_source_once("quat", "base_quat_measured")
            return arr[:4]

    if "base_quat_target" in proprio:
        arr = np.asarray(proprio["base_quat_target"], dtype=np.float64).reshape(-1)
        if arr.size >= 4:
            _set_source_once("quat", "base_quat_target")
            return arr[:4]

    if "floating_base_pose" in proprio:
        arr = np.asarray(proprio["floating_base_pose"], dtype=np.float64).reshape(-1)
        if arr.size >= 7:
            _set_source_once("quat", "floating_base_pose[3:7]")
            return arr[3:7]

    if "base_pose" in proprio:
        arr = np.asarray(proprio["base_pose"], dtype=np.float64).reshape(-1)
        if arr.size >= 7:
            _set_source_once("quat", "base_pose[3:7]")
            return arr[3:7]

    odo_state = proprio.get("odo_state")
    if isinstance(odo_state, dict) and "orientation" in odo_state:
        arr = np.asarray(odo_state["orientation"], dtype=np.float64).reshape(-1)
        if arr.size >= 4:
            _set_source_once("quat", "odo_state.orientation")
            return arr[:4]

    _set_source_once("quat", "default_identity")
    return _DEFAULT_BASE_QUAT.copy()


class GrootDataCollectorWithBasePose(GrootDataCollector):
    def _add_cpp_state_features(self, frame_data: dict, proprio: dict) -> None:
        super()._add_cpp_state_features(frame_data, proprio)
        frame_data["robot.base_pos"] = _extract_base_pos(proprio)
        frame_data["robot.base_quat"] = _extract_base_quat(proprio)


def main(config: SonicDataExporterConfig) -> None:
    g1_rm = get_g1_robot_model()

    dataset_features = _get_features_with_base_pose(g1_rm)
    modality_config = _get_modality_config_with_base_pose(g1_rm)

    if config.record_wrist_cameras:
        print("[Camera] Wrist cameras enabled — adding to dataset schema")
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
            "record_wrist_cameras": config.record_wrist_cameras,
            "records_base_pose": True,
        },
    )

    data_collector = GrootDataCollectorWithBasePose(
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
    )
    data_collector.run()


if __name__ == "__main__":
    config = tyro.cli(SonicDataExporterConfig)
    if config.dataset_name is None:
        config.dataset_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    main(config)
