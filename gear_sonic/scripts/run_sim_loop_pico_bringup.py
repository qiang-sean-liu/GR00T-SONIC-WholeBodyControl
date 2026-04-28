"""Launch the pico_bringup simulator loop without modifying this branch.

This wrapper loads the pico_bringup versions of:
- gear_sonic/scripts/run_sim_loop.py
- gear_sonic/utils/mujoco_sim/{configs,base_sim,simulator_factory}.py

It also materializes the pico_bringup pnp_cube XML + head-camera-enabled G1 XML
into a temporary directory and patches CubeEnv to use that scene.

Example:
    python gear_sonic/scripts/run_sim_loop_pico_bringup.py \
      --env_name pnp_cube \
      --head_cam \
      --enable_image_publish \
      --enable_offscreen
"""

from __future__ import annotations

import atexit
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path


BRANCH = "pico_bringup"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_show(path_in_repo: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{BRANCH}:{path_in_repo}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _load_branch_module(module_name: str, path_in_repo: str) -> types.ModuleType:
    source = _git_show(path_in_repo)
    module = types.ModuleType(module_name)
    module.__file__ = str(REPO_ROOT / path_in_repo)
    module.__package__ = module_name.rpartition(".")[0]
    sys.modules[module_name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _prepare_temp_xmls() -> tuple[Path, Path, str]:
    temp_dir = Path(tempfile.mkdtemp(prefix="pico_bringup_sim_"))
    atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

    g1_xml = temp_dir / "g1_29dof_with_hand_rev_1_0_activatedfinger.xml"
    pnp_xml = temp_dir / "pnp_cube_43dof.xml"
    meshes_link = temp_dir / "meshes"

    g1_xml.write_text(
        _git_show(
            "decoupled_wbc/control/robot_model/model_data/g1/"
            "g1_29dof_with_hand_rev_1_0_activatedfinger.xml"
        ),
        encoding="utf-8",
    )
    pnp_xml.write_text(
        _git_show("decoupled_wbc/control/robot_model/model_data/g1/pnp_cube_43dof.xml"),
        encoding="utf-8",
    )

    source_meshes = (
        REPO_ROOT / "decoupled_wbc/control/robot_model/model_data/g1/meshes"
    )
    if not source_meshes.exists():
        raise FileNotFoundError(f"Expected meshes directory not found: {source_meshes}")
    meshes_link.symlink_to(source_meshes, target_is_directory=True)

    return temp_dir, g1_xml, str(pnp_xml)


def main() -> int:
    _, _, pnp_cube_xml = _prepare_temp_xmls()

    _load_branch_module(
        "gear_sonic.utils.mujoco_sim.configs",
        "gear_sonic/utils/mujoco_sim/configs.py",
    )
    base_sim = _load_branch_module(
        "gear_sonic.utils.mujoco_sim.base_sim",
        "gear_sonic/utils/mujoco_sim/base_sim.py",
    )
    _load_branch_module(
        "gear_sonic.utils.mujoco_sim.simulator_factory",
        "gear_sonic/utils/mujoco_sim/simulator_factory.py",
    )

    def _cube_env_init(self, config, **kwargs):
        config = config.copy()
        config["ROBOT_SCENE"] = pnp_cube_xml
        base_sim.DefaultEnv.__init__(self, config, "pnp_cube", **kwargs)

    base_sim.CubeEnv.__init__ = _cube_env_init

    run_sim_loop_src = _git_show("gear_sonic/scripts/run_sim_loop.py")
    namespace = {
        "__name__": "__main__",
        "__file__": str(REPO_ROOT / "gear_sonic/scripts/run_sim_loop.py"),
        "__package__": None,
    }
    exec(compile(run_sim_loop_src, namespace["__file__"], "exec"), namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
