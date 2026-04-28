"""Run MuJoCo and publish live floating-base pose for data collection.

This is the same simulator entrypoint as ``run_sim_loop.py`` plus a ZMQ
``base_state`` publisher. The data exporter subscribes to this topic on port
5558 and records the live MuJoCo root pose as ``robot.base_pos`` and
``robot.base_quat``.
"""

from dataclasses import dataclass
import json
import pickle
import time
from typing import Any, Dict

import numpy as np
import tyro
import zmq

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.data.robot_model.robot_model import RobotModel
from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel


@dataclass
class SimLoopWithBaseStateConfig(SimLoopConfig):
    base_state_zmq_bind: str = "*"
    """Address to bind the base_state publisher."""

    base_state_zmq_port: int = 5558
    """ZMQ PUB port for MuJoCo base_state messages."""

    base_state_publish_frequency: int = 100
    """Frequency, in Hz, for publishing MuJoCo base pose."""


ArgsConfig = SimLoopWithBaseStateConfig


class BaseStatePublishingSimulator(BaseSimulator):
    def __init__(
        self,
        *args,
        base_state_zmq_bind: str = "*",
        base_state_zmq_port: int = 5558,
        base_state_publish_frequency: int = 100,
        **kwargs,
    ):
        self._base_state_zmq_bind = base_state_zmq_bind
        self._base_state_zmq_port = base_state_zmq_port
        self._base_state_publish_period = 1.0 / max(base_state_publish_frequency, 1)
        self._base_state_last_publish_time = 0.0
        self._base_state_zmq_ctx = None
        self._base_state_zmq_socket = None
        super().__init__(*args, **kwargs)

    def init_publisher(self):
        super().init_publisher()
        if self._base_state_zmq_port <= 0:
            return

        endpoint = f"tcp://{self._base_state_zmq_bind}:{self._base_state_zmq_port}"
        self._base_state_zmq_ctx = zmq.Context()
        self._base_state_zmq_socket = self._base_state_zmq_ctx.socket(zmq.PUB)
        self._base_state_zmq_socket.setsockopt(zmq.SNDHWM, 5)
        self._base_state_zmq_socket.setsockopt(zmq.LINGER, 0)
        self._base_state_zmq_socket.bind(endpoint)
        print(f"[BaseState] Publishing MuJoCo base_state on {endpoint}")

    def _publish_base_state(self, now: float) -> None:
        if self._base_state_zmq_socket is None:
            return
        if now - self._base_state_last_publish_time < self._base_state_publish_period:
            return

        qpos = np.asarray(self.sim_env.mj_data.qpos, dtype=np.float64)
        payload = {
            "base_pos_sim": qpos[:3].tolist(),
            "base_quat_sim": qpos[3:7].tolist(),  # MuJoCo free-joint quat is [w, x, y, z].
            "sim_time": float(self.sim_env.mj_data.time),
            "wall_time": now,
        }
        raw = b"base_state" + json.dumps(payload, separators=(",", ":")).encode("utf-8")
        try:
            self._base_state_zmq_socket.send(raw, flags=zmq.NOBLOCK)
            self._base_state_last_publish_time = now
        except zmq.Again:
            pass

    def start(self):
        """Main simulation loop with live base_state publishing."""
        sim_cnt = 0
        ts = time.time()

        try:
            while self._running and (
                (self.sim_env.viewer and self.sim_env.viewer.is_running())
                or (self.sim_env.viewer is None)
            ):
                step_start = time.monotonic()

                self.sim_env.sim_step()
                now = time.time()
                self._publish_base_state(now)

                if now - ts > 1 / 10.0 and self.redis_client is not None:
                    head_pose = self.sim_env.get_head_pose()
                    self.redis_client.set("head_pos", pickle.dumps(head_pose[:3]))
                    self.redis_client.set("head_quat", pickle.dumps(head_pose[3:]))
                    ts = now

                if sim_cnt % int(self.viewer_dt / self.sim_dt) == 0:
                    self.sim_env.update_viewer()

                if sim_cnt % int(self.reward_dt / self.sim_dt) == 0:
                    self.sim_env.update_reward()

                if sim_cnt % int(self.image_dt / self.sim_dt) == 0:
                    self.sim_env.update_render_caches()

                elapsed = time.monotonic() - step_start
                sleep_time = self.sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                sim_cnt += 1
        except KeyboardInterrupt:
            print("Simulator interrupted by user.")
        finally:
            self.close()

    def close(self):
        try:
            if self._base_state_zmq_socket is not None:
                self._base_state_zmq_socket.close()
            if self._base_state_zmq_ctx is not None:
                self._base_state_zmq_ctx.term()
        finally:
            self._base_state_zmq_socket = None
            self._base_state_zmq_ctx = None
            super().close()


class SimWrapper:
    def __init__(
        self,
        robot_model: RobotModel,
        env_name: str,
        config: Dict[str, Any],
        base_state_zmq_bind: str,
        base_state_zmq_port: int,
        base_state_publish_frequency: int,
        **kwargs,
    ):
        self.robot_model = robot_model
        self.config = config

        init_channel(config=self.config)

        self.sim = BaseStatePublishingSimulator(
            config=self.config,
            env_name=env_name,
            base_state_zmq_bind=base_state_zmq_bind,
            base_state_zmq_port=base_state_zmq_port,
            base_state_publish_frequency=base_state_publish_frequency,
            **kwargs,
        )


def main(config: ArgsConfig):
    wbc_config = config.load_wbc_yaml()
    wbc_config["ENV_NAME"] = config.env_name

    if config.enable_image_publish:
        assert (
            config.enable_offscreen
        ), "enable_offscreen must be True when enable_image_publish is True"

    robot_model = instantiate_g1_robot_model()

    sim_wrapper = SimWrapper(
        robot_model=robot_model,
        env_name=config.env_name,
        config=wbc_config,
        base_state_zmq_bind=config.base_state_zmq_bind,
        base_state_zmq_port=config.base_state_zmq_port,
        base_state_publish_frequency=config.base_state_publish_frequency,
        onscreen=wbc_config.get("ENABLE_ONSCREEN", True),
        offscreen=wbc_config.get("ENABLE_OFFSCREEN", False),
        enable_image_publish=config.enable_image_publish,
    )

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
