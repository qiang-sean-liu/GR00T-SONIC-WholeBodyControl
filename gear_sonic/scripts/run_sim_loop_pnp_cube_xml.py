"""Run MuJoCo using the imported pico_bringup pnp_cube XML.

This script is intentionally separate from `run_sim_loop.py` and only overrides
the robot scene path. It does not add head-camera behavior.
"""

from typing import Dict

import tyro

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.data.robot_model.robot_model import RobotModel
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel

ArgsConfig = SimLoopConfig

PNP_CUBE_SCENE = (
    "decoupled_wbc/control/robot_model/model_data/g1/"
    "pnp_cube_pico_bringup_43dof.xml"
)


class SimWrapper:
    def __init__(self, robot_model: RobotModel, config: Dict[str, any], **kwargs):
        self.robot_model = robot_model
        self.config = config

        init_channel(config=self.config)
        self.sim = SimulatorFactory.create_simulator(
            config=self.config,
            env_name="default",
            **kwargs,
        )


def main(config: ArgsConfig):
    wbc_config = config.load_wbc_yaml()
    wbc_config["ENV_NAME"] = "pnp_cube"
    wbc_config["ROBOT_SCENE"] = PNP_CUBE_SCENE

    if config.enable_image_publish:
        assert config.enable_offscreen, "enable_offscreen must be True when enable_image_publish is True"

    robot_model = instantiate_g1_robot_model()
    sim_wrapper = SimWrapper(
        robot_model=robot_model,
        config=wbc_config,
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
