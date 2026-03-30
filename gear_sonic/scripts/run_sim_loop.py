"""Entry point for running a MuJoCo simulation loop with the G1 robot model.

Parses a YAML-based WBC config via tyro CLI, instantiates the G1 robot model,
and launches the simulator (optionally with offscreen image publishing).
"""

from typing import Dict

import tyro

from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.data.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from gear_sonic.data.robot_model.robot_model import RobotModel

ArgsConfig = SimLoopConfig


class SimWrapper:
    def __init__(self, robot_model: RobotModel, env_name: str, config: Dict[str, any], **kwargs):
        self.robot_model = robot_model
        self.config = config

        init_channel(config=self.config)

        # Create simulator using factory
        self.sim = SimulatorFactory.create_simulator(
            config=self.config,
            env_name=env_name,
            **kwargs,
        )


def main(config: ArgsConfig):
    wbc_config = config.load_wbc_yaml()
    # NOTE: we will override the interface to local if it is not specified
    wbc_config["ENV_NAME"] = config.env_name

    if config.enable_image_publish:
        assert (
            config.enable_offscreen
        ), "enable_offscreen must be True when enable_image_publish is True"

    robot_model = instantiate_g1_robot_model()

    # --head_cam: render stereo head cameras offscreen and write a side-by-side
    # (left | right) frame to named shared memory for streaming to PICO via
    # stream_cam_xr.py (XRoboToolkit Remote Vision panel).
    # Each eye: 640×480; shared memory frame: 1280×480.
    head_cam_kwargs = {}
    if config.head_cam:
        head_cam_kwargs = {
            "camera_configs": {
                "head_camera_left":  {"height": 480, "width": 640},
                "head_camera_right": {"height": 480, "width": 640},
            },
            "head_cam_shm_name": "pico_head_cam",
        }

    # --enable_image_publish: ZMQ-based image publishing (data collection / analysis).
    # Also needs camera_configs if not already set by --head_cam.
    if config.enable_image_publish and not config.head_cam:
        head_cam_kwargs["camera_configs"] = {"head_camera": {"height": 480, "width": 640}}

    sim_wrapper = SimWrapper(
        robot_model=robot_model,
        env_name=config.env_name,
        config=wbc_config,
        onscreen=wbc_config.get("ENABLE_ONSCREEN", True),
        offscreen=wbc_config.get("ENABLE_OFFSCREEN", False) or config.head_cam or config.enable_image_publish,
        enable_image_publish=config.enable_image_publish,
        **head_cam_kwargs,
    )
    # Start simulator as independent process
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