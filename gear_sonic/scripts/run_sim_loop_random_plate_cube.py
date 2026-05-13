"""Run the default MuJoCo scene with randomized plate/cube resets.

The default scene already contains ``plate_body`` and ``cube_body`` in
``gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml``. This launcher
keeps that scene unchanged and only patches reset behavior:

- Backspace still resets the simulation through ``DefaultEnv.reset``.
- After every reset, only the plate and cube table-top positions are randomized.
- Robot, table, cameras, and all other scene elements remain at their default
  positions.

Example:
    python gear_sonic/scripts/run_sim_loop_random_plate_cube.py \
      --enable-image-publish \
      --enable-offscreen \
      --camera-port 5555
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
import tyro

from gear_sonic.scripts.run_sim_loop import ArgsConfig, main
from gear_sonic.utils.mujoco_sim import base_sim


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_XML_PATH = REPO_ROOT / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
RUNTIME_SCENE_XML_PATH = REPO_ROOT / "outputs/latest_random_plate_cube_scene.xml"

# Default scene table geometry:
# table_body pos=(1.5, 0, 0.4), table_top local pos=(0, 0, 0.3),
# table_top size=(0.35, 0.7, 0.05). Top surface z = 0.75.
TABLE_X_RANGE = (1.234, 1.766)
TABLE_Y_RANGE = (-0.504, 0.504)
TABLE_TOP_Z = 0.75
CUBE_HALF_SIZE = 0.035
PLATE_RADIUS = 0.12
PLATE_HALF_HEIGHT = 0.01
TABLE_CLEARANCE = 0.002
MIN_PLATE_CUBE_CLEARANCE = 0.08
MIN_CENTER_DISTANCE = PLATE_RADIUS + np.sqrt(2.0) * CUBE_HALF_SIZE + MIN_PLATE_CUBE_CLEARANCE
MIN_REPEAT_DISTANCE = 0.08


def _sample_positions(
    rng: np.random.Generator,
    previous: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample valid separated table-top positions for plate and cube."""
    for _ in range(100):
        plate_xy = np.array(
            [
                rng.uniform(TABLE_X_RANGE[0] + PLATE_RADIUS, TABLE_X_RANGE[1] - PLATE_RADIUS),
                rng.uniform(TABLE_Y_RANGE[0] + PLATE_RADIUS, TABLE_Y_RANGE[1] - PLATE_RADIUS),
            ],
            dtype=np.float64,
        )
        cube_xy = np.array(
            [
                rng.uniform(TABLE_X_RANGE[0] + CUBE_HALF_SIZE, TABLE_X_RANGE[1] - CUBE_HALF_SIZE),
                rng.uniform(TABLE_Y_RANGE[0] + CUBE_HALF_SIZE, TABLE_Y_RANGE[1] - CUBE_HALF_SIZE),
            ],
            dtype=np.float64,
        )

        if np.linalg.norm(plate_xy - cube_xy) < MIN_CENTER_DISTANCE:
            continue

        plate_pos = np.array(
            [plate_xy[0], plate_xy[1], TABLE_TOP_Z + PLATE_HALF_HEIGHT + TABLE_CLEARANCE],
            dtype=np.float64,
        )
        cube_pos = np.array(
            [cube_xy[0], cube_xy[1], TABLE_TOP_Z + CUBE_HALF_SIZE + TABLE_CLEARANCE],
            dtype=np.float64,
        )

        if previous is not None:
            prev_plate, prev_cube = previous
            plate_delta = np.linalg.norm(plate_pos[:2] - prev_plate[:2])
            cube_delta = np.linalg.norm(cube_pos[:2] - prev_cube[:2])
            if plate_delta < MIN_REPEAT_DISTANCE and cube_delta < MIN_REPEAT_DISTANCE:
                continue

        return plate_pos, cube_pos

    raise RuntimeError("Failed to sample separated plate/cube positions")


def _format_pos(pos: np.ndarray) -> str:
    return f"{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}"


def _write_runtime_scene_xml(plate_pos: np.ndarray, cube_pos: np.ndarray) -> None:
    """Write a source XML snapshot with the latest randomized object poses."""
    tree = ET.parse(DEFAULT_SCENE_XML_PATH)
    root = tree.getroot()

    for include in root.findall("include"):
        include_file = include.get("file")
        if include_file:
            include.set("file", str((DEFAULT_SCENE_XML_PATH.parent / include_file).resolve()))

    plate_body = root.find(".//body[@name='plate_body']")
    cube_body = root.find(".//body[@name='cube_body']")
    if plate_body is None or cube_body is None:
        raise RuntimeError("Default scene XML is missing plate_body or cube_body")

    plate_body.set("pos", _format_pos(plate_pos))
    cube_body.set("pos", _format_pos(cube_pos))

    RUNTIME_SCENE_XML_PATH.parent.mkdir(parents=True, exist_ok=True)
    tree.write(RUNTIME_SCENE_XML_PATH, encoding="utf-8", xml_declaration=True)


def _install_randomized_default_reset() -> None:
    original_init = base_sim.DefaultEnv.__init__
    original_reset = base_sim.DefaultEnv.reset
    original_sim_step = base_sim.DefaultEnv.sim_step
    original_launch_passive = mujoco.viewer.launch_passive
    active_env: dict[str, object | None] = {"env": None}

    key_names = {
        259: "backspace",
        86: "v",
        265: "up",
        264: "down",
        263: "left",
        262: "right",
    }

    def _launch_passive_with_random_reset(*args, key_callback=None, **kwargs):
        def _key_callback(keycode):
            if key_callback is not None:
                key_callback(keycode)

            env = active_env.get("env")
            key_name = key_names.get(int(keycode))
            if env is not None and key_name == "backspace":
                # MuJoCo callbacks run from the viewer side. Queue the reset and
                # let the simulation loop mutate mj_model/mj_data.
                env._random_plate_cube_reset_requested = True
            elif env is not None and key_name is not None:
                env.handle_keyboard_button(key_name)

        return original_launch_passive(*args, key_callback=_key_callback, **kwargs)

    def _init_randomized_scene(self, *args, **kwargs):
        active_env["env"] = self
        original_init(self, *args, **kwargs)

        self._random_plate_cube_rng = np.random.default_rng()
        self._random_plate_cube_previous = None
        self._random_plate_cube_reset_count = 0
        self._random_plate_cube_reset_requested = False
        self._random_plate_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "plate_body"
        )
        self._random_cube_joint_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
        )
        self._random_cube_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube_body"
        )

        if self._random_plate_body_id < 0:
            raise RuntimeError("plate_body not found in default scene")
        if self._random_cube_joint_id < 0:
            raise RuntimeError("cube_joint not found in default scene")
        if self._random_cube_body_id < 0:
            raise RuntimeError("cube_body not found in default scene")

        self._random_cube_qpos_adr = int(self.mj_model.jnt_qposadr[self._random_cube_joint_id])
        self._random_cube_qvel_adr = int(self.mj_model.jnt_dofadr[self._random_cube_joint_id])
        if self.viewer is not None:
            with self.viewer.lock():
                self.randomize_plate_and_cube()
        else:
            self.randomize_plate_and_cube()

    def _randomize_plate_and_cube(self) -> None:
        plate_pos, cube_pos = _sample_positions(
            self._random_plate_cube_rng,
            self._random_plate_cube_previous,
        )

        # plate_body is static in the model; cube_body is driven by its free joint qpos.
        self.mj_model.body_pos[self._random_plate_body_id] = plate_pos
        self.mj_model.qpos0[self._random_cube_qpos_adr : self._random_cube_qpos_adr + 3] = cube_pos
        self.mj_model.qpos0[
            self._random_cube_qpos_adr + 3 : self._random_cube_qpos_adr + 7
        ] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.mj_data.qpos[self._random_cube_qpos_adr : self._random_cube_qpos_adr + 3] = cube_pos
        self.mj_data.qpos[self._random_cube_qpos_adr + 3 : self._random_cube_qpos_adr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self.mj_data.qvel[self._random_cube_qvel_adr : self._random_cube_qvel_adr + 6] = 0.0
        self.mj_data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self._random_plate_cube_previous = (plate_pos.copy(), cube_pos.copy())
        self._random_plate_cube_reset_count += 1
        actual_plate_pos = self.mj_data.xpos[self._random_plate_body_id].copy()
        actual_cube_pos = self.mj_data.xpos[self._random_cube_body_id].copy()
        _write_runtime_scene_xml(actual_plate_pos, actual_cube_pos)
        print(
            "[RandomPlateCube] reset "
            f"{self._random_plate_cube_reset_count}: "
            f"plate=({actual_plate_pos[0]:.3f}, {actual_plate_pos[1]:.3f}, {actual_plate_pos[2]:.3f}) "
            f"cube=({actual_cube_pos[0]:.3f}, {actual_cube_pos[1]:.3f}, {actual_cube_pos[2]:.3f})"
        )

    def _reset_randomized_scene(self):
        if self.viewer is not None:
            with self.viewer.lock():
                original_reset(self)
                self.randomize_plate_and_cube()
        else:
            original_reset(self)
            self.randomize_plate_and_cube()

    def _sim_step_randomized_scene(self):
        if getattr(self, "_random_plate_cube_reset_requested", False):
            self._random_plate_cube_reset_requested = False
            self.reset()
        original_sim_step(self)

    base_sim.DefaultEnv.__init__ = _init_randomized_scene
    base_sim.DefaultEnv.randomize_plate_and_cube = _randomize_plate_and_cube
    base_sim.DefaultEnv.reset = _reset_randomized_scene
    base_sim.DefaultEnv.sim_step = _sim_step_randomized_scene
    mujoco.viewer.launch_passive = _launch_passive_with_random_reset


if __name__ == "__main__":
    _install_randomized_default_reset()
    config = tyro.cli(ArgsConfig)
    main(config)
