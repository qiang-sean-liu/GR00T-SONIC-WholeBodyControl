"""Sonic VLA exporter with lowcmd debug and exact SONIC ONNX buffers.

This recovers the ONNX-input debug exporter used for datasets such as
``outputs/onnx-inputs-debug``.  It extends ``run_data_exporter_lowcmd.py`` with
the C++ deploy buffers needed to verify encoder/decoder playback:

  - robot.base_ang_vel
  - robot.body_dq
  - sonic.encoder_obs
  - sonic.token_state
  - sonic.decoder_obs
  - sonic.decoder_action_raw
  - sonic.q_target_cmd
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import time

import msgpack
import msgpack_numpy as mnp
import numpy as np
import tyro
import zmq

from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.features_sonic_vla import (
    get_features_sonic_vla,
    get_g1_robot_model,
    get_modality_config_sonic_vla,
    get_wrist_camera_features,
    get_wrist_camera_modality_config,
)
from gear_sonic.scripts.run_data_exporter import (
    SonicDataExporterConfig,
    TextToSpeech,
    poll_robot_config_zmq,
)
from gear_sonic.scripts.run_data_exporter_lowcmd import (
    LowCmdDebugDataCollector,
    _add_array_feature,
    _add_lowcmd_debug_features,
    _add_scalar_feature,
)
from gear_sonic.utils.data_collection.zmq_state_subscriber import STATE_ZMQ_TOPIC

_BODY_DQ_DIM = 29
_BASE_ANG_VEL_DIM = 3
_ENCODER_OBS_DIM = 1762
_TOKEN_DIM = 64
_DECODER_OBS_DIM = 994
_DECODER_ACTION_DIM = 29
_Q_TARGET_CMD_DIM = 29

_LEGACY_DEBUG_ARRAY_FEATURES: dict[str, int] = {
    "robot.motor_tau_est": 29,
    "robot.left_hand_motor_tau_est": 7,
    "robot.right_hand_motor_tau_est": 7,
    "robot.motor_pd_torque": 29,
    "robot.motor_pd_torque_raw": 29,
    "robot.motor_pd_q": 29,
    "robot.motor_pd_dq": 29,
    "robot.motor_pd_q_des": 29,
    "robot.motor_pd_dq_des": 29,
    "robot.motor_pd_kp": 29,
    "robot.motor_pd_kd": 29,
    "robot.motor_pd_tau_ff": 29,
    "robot.left_hand_motor_pd_torque": 7,
    "robot.right_hand_motor_pd_torque": 7,
    "robot.motor_pd_substep_q": 116,
    "robot.motor_pd_substep_dq": 116,
    "robot.motor_pd_substep_q_des": 116,
    "robot.motor_pd_substep_dq_des": 116,
    "robot.motor_pd_substep_kp": 116,
    "robot.motor_pd_substep_kd": 116,
    "robot.motor_pd_substep_tau_ff": 116,
    "robot.motor_pd_substep_torque_raw": 116,
    "robot.motor_pd_substep_torque": 116,
    "robot.motor_pd_substep_sim_time": 4,
    "robot.motor_pd_substep_wall_time": 4,
    "robot.mujoco_substep_qpos": 228,
    "robot.mujoco_substep_qvel": 220,
    "robot.mujoco_substep_ctrl": 172,
    "robot.mujoco_substep_qfrc_applied": 220,
    "robot.mujoco_substep_xfrc_applied": 1152,
    "robot.mujoco_substep_qacc_warmstart": 220,
    "robot.mujoco_substep_post_qpos": 228,
    "robot.mujoco_substep_post_qvel": 220,
    "robot.mujoco_substep_post_qacc": 220,
    "robot.mujoco_substep_post_actuator_force": 172,
    "robot.mujoco_substep_post_sim_time": 4,
    "robot.mujoco_substep_post_wall_time": 4,
}
_LEGACY_DEBUG_SCALAR_FEATURES: tuple[str, ...] = (
    "robot.observation_state_source_timestamp",
    "robot.observation_state_receive_timestamp",
    "robot.zmq_send_timestamp",
    "robot.zmq_receive_timestamp",
    "robot.record_timestamp",
)

_FRAME_RATE_CHECK_SECONDS = 2.0
_FRAME_RATE_CHECK_TIMEOUT_SECONDS = 5.0
_FRAME_RATE_WARN_REL_TOL = 0.15


def _decode_state_message(raw: bytes, topic: str) -> dict:
    payload = raw[len(topic):]
    return msgpack.unpackb(payload, raw=False)


def check_state_frame_rate(host: str, port: int, expected_hz: float) -> dict:
    """Sample g1_debug at startup and report whether publisher Hz is plausible."""
    mnp.patch()
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, STATE_ZMQ_TOPIC)
    sub.setsockopt(zmq.RCVTIMEO, 200)
    sub.setsockopt(zmq.RCVHWM, 1000)
    sub.connect(f"tcp://{host}:{port}")

    print(
        f"[FrameRate] Checking {STATE_ZMQ_TOPIC} on tcp://{host}:{port} "
        f"for {_FRAME_RATE_CHECK_SECONDS:.1f}s ..."
    )
    time.sleep(0.2)

    receive_times: list[float] = []
    send_times: list[float] = []
    first_msg_deadline = time.monotonic() + _FRAME_RATE_CHECK_TIMEOUT_SECONDS
    sample_start: float | None = None

    try:
        while True:
            now = time.monotonic()
            if sample_start is None and now >= first_msg_deadline:
                print(
                    f"[FrameRate] WARNING: no {STATE_ZMQ_TOPIC} messages received "
                    f"within {_FRAME_RATE_CHECK_TIMEOUT_SECONDS:.1f}s."
                )
                return {
                    "state_fps_check_expected_hz": float(expected_hz),
                    "state_fps_check_measured_hz": float("nan"),
                    "state_fps_check_message_count": 0,
                    "state_fps_check_ok": False,
                }
            if sample_start is not None and now - sample_start >= _FRAME_RATE_CHECK_SECONDS:
                break

            try:
                raw = sub.recv()
            except zmq.Again:
                continue

            msg_time = time.monotonic()
            if sample_start is None:
                sample_start = msg_time
            receive_times.append(msg_time)

            try:
                msg = _decode_state_message(raw, STATE_ZMQ_TOPIC)
                send_ts = msg.get("zmq_send_timestamp")
                if send_ts is not None:
                    arr = np.asarray(send_ts, dtype=np.float64).reshape(-1)
                    if arr.size and np.isfinite(arr[0]):
                        send_times.append(float(arr[0]))
            except Exception:
                pass
    finally:
        sub.close()
        ctx.term()

    count = len(receive_times)
    elapsed = receive_times[-1] - receive_times[0] if count >= 2 else 0.0
    measured_hz = (count - 1) / elapsed if elapsed > 0 else float("nan")
    dt = np.diff(receive_times) if count >= 2 else np.array([], dtype=np.float64)
    dt_mean_ms = float(np.mean(dt) * 1000.0) if dt.size else float("nan")
    dt_max_ms = float(np.max(dt) * 1000.0) if dt.size else float("nan")
    ok = bool(
        np.isfinite(measured_hz)
        and abs(measured_hz - expected_hz) <= expected_hz * _FRAME_RATE_WARN_REL_TOL
    )
    status = "OK" if ok else "WARNING"
    print(
        f"[FrameRate] {status}: received {count} messages, "
        f"measured {measured_hz:.2f} Hz (expected {expected_hz:.2f} Hz), "
        f"dt_mean={dt_mean_ms:.2f} ms, dt_max={dt_max_ms:.2f} ms"
    )

    result = {
        "state_fps_check_expected_hz": float(expected_hz),
        "state_fps_check_measured_hz": float(measured_hz),
        "state_fps_check_message_count": int(count),
        "state_fps_check_dt_mean_ms": dt_mean_ms,
        "state_fps_check_dt_max_ms": dt_max_ms,
        "state_fps_check_ok": ok,
    }

    if len(send_times) >= 2:
        send_dt = np.diff(np.asarray(send_times, dtype=np.float64))
        send_dt = send_dt[np.isfinite(send_dt) & (send_dt > 0)]
        if send_dt.size:
            result["state_fps_check_sender_hz"] = float(1.0 / np.mean(send_dt))
            result["state_fps_check_sender_dt_max_ms"] = float(np.max(send_dt) * 1000.0)
            print(
                "[FrameRate] Publisher timestamp rate: "
                f"{result['state_fps_check_sender_hz']:.2f} Hz, "
                f"max sender dt={result['state_fps_check_sender_dt_max_ms']:.2f} ms"
            )

    return result


def _add_onnx_input_features(features: dict) -> None:
    _add_array_feature(features, "robot.base_ang_vel", _BASE_ANG_VEL_DIM)
    _add_array_feature(features, "robot.body_dq", _BODY_DQ_DIM)
    _add_array_feature(features, "sonic.encoder_obs", _ENCODER_OBS_DIM)
    _add_array_feature(features, "sonic.token_state", _TOKEN_DIM)
    _add_array_feature(features, "sonic.decoder_obs", _DECODER_OBS_DIM)
    _add_array_feature(features, "sonic.decoder_action_raw", _DECODER_ACTION_DIM)
    _add_array_feature(features, "sonic.q_target_cmd", _Q_TARGET_CMD_DIM)


def _add_legacy_onnx_debug_features(features: dict) -> None:
    """Match the schema used by the existing onnx-inputs-debug dataset."""
    for key, length in _LEGACY_DEBUG_ARRAY_FEATURES.items():
        _add_array_feature(features, key, length)
    for key in _LEGACY_DEBUG_SCALAR_FEATURES:
        _add_scalar_feature(features, key)


class OnnxInputsDataCollector(LowCmdDebugDataCollector):
    """Collector that appends exact SONIC ONNX model inputs/outputs."""

    def _add_cpp_state_features(self, frame_data: dict, proprio: dict) -> None:
        super()._add_cpp_state_features(frame_data, proprio)
        self._add_legacy_onnx_debug_frame_features(frame_data, proprio)
        self._add_onnx_input_frame_features(frame_data, proprio)
        self._fill_missing_robot_sonic_features(frame_data)

    def _source_for_debug(self, proprio: dict) -> dict:
        if self.latest_base_state_msg is not None:
            return {**proprio, **self.latest_base_state_msg}
        return proprio

    def _add_legacy_onnx_debug_frame_features(self, frame_data: dict, proprio: dict) -> None:
        source = self._source_for_debug(proprio)
        for output_key, length in _LEGACY_DEBUG_ARRAY_FEATURES.items():
            if output_key in frame_data:
                continue
            source_key = output_key.removeprefix("robot.")
            frame_data[output_key] = self._array_from_sources(
                source,
                (source_key, output_key),
                output_key,
                length,
            )

        for output_key in _LEGACY_DEBUG_SCALAR_FEATURES:
            if output_key in frame_data:
                continue
            source_key = output_key.removeprefix("robot.")
            if output_key == "robot.record_timestamp":
                frame_data[output_key] = np.array([time.time()], dtype=np.float64)
            else:
                frame_data[output_key] = self._scalar_from_sources(
                    source,
                    (source_key, output_key),
                    output_key,
                )

    def _fill_missing_robot_sonic_features(self, frame_data: dict) -> None:
        """Keep appends compatible with existing datasets that have a wider schema."""
        for key, spec in self.data_exporter.features.items():
            if key in frame_data or not key.startswith(("robot.", "sonic.")):
                continue
            shape = tuple(spec.get("shape", ()))
            dtype = spec.get("dtype", "float64")
            fill_value = np.nan if str(dtype).startswith("float") else 0
            frame_data[key] = np.full(shape, fill_value, dtype=np.dtype(dtype))

    def _add_onnx_input_frame_features(self, frame_data: dict, proprio: dict) -> None:
        sim_source = (
            {**proprio, **self.latest_base_state_msg}
            if self.latest_base_state_msg is not None
            else proprio
        )

        frame_data["robot.base_ang_vel"] = self._array_from_sources(
            sim_source,
            ("base_ang_vel", "robot.base_ang_vel", "secondary_imu_vel"),
            "robot.base_ang_vel",
            _BASE_ANG_VEL_DIM,
        )
        frame_data["robot.body_dq"] = self._array_from_sources(
            sim_source,
            ("body_dq", "robot.body_dq", "motor_dq_measured"),
            "robot.body_dq",
            _BODY_DQ_DIM,
        )
        frame_data["sonic.encoder_obs"] = self._array_from_sources(
            proprio,
            ("encoder_obs", "sonic.encoder_obs"),
            "sonic.encoder_obs",
            _ENCODER_OBS_DIM,
        )
        frame_data["sonic.token_state"] = self._array_from_sources(
            proprio,
            ("token_state", "sonic.token_state", "token"),
            "sonic.token_state",
            _TOKEN_DIM,
        )
        frame_data["sonic.decoder_obs"] = self._array_from_sources(
            proprio,
            ("decoder_obs", "sonic.decoder_obs"),
            "sonic.decoder_obs",
            _DECODER_OBS_DIM,
        )
        frame_data["sonic.decoder_action_raw"] = self._array_from_sources(
            proprio,
            ("decoder_action_raw", "sonic.decoder_action_raw", "raw_action"),
            "sonic.decoder_action_raw",
            _DECODER_ACTION_DIM,
        )
        frame_data["sonic.q_target_cmd"] = self._array_from_sources(
            proprio,
            ("q_target_cmd", "sonic.q_target_cmd", "body_motor_cmd_q", "motor_q"),
            "sonic.q_target_cmd",
            _Q_TARGET_CMD_DIM,
        )


def main(config: SonicDataExporterConfig) -> None:
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
    _add_legacy_onnx_debug_features(dataset_features)
    _add_onnx_input_features(dataset_features)

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
    frame_rate_check = check_state_frame_rate(
        config.state_zmq_host,
        config.state_zmq_port,
        float(config.data_collection_frequency),
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
            **frame_rate_check,
            "records_base_pos": True,
            "records_lowcmd_debug": True,
            "records_onnx_inputs": True,
            "onnx_inputs_note": (
                "NaN means the corresponding ZMQ publisher field was not present."
            ),
        },
    )

    data_collector = OnnxInputsDataCollector(
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
        cfg.dataset_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-onnx-inputs-debug")
    main(cfg)
