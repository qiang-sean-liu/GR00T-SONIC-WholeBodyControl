"""A/B compare SONIC replay with recorded SMPL encoder blocks.

This script reuses playback_lerobot SonicRunner logic and reports body-joint L2
metrics for:
  1) baseline_rebuild: rebuild encoder inputs from dataset signals
  2) inject_recorded_smpl_blocks: rebuild encoder obs, then overwrite only:
       - smpl_joints_10frame_step1
       - smpl_anchor_orientation_10frame_step1
       - motion_joint_positions_wrists_10frame_step1
  3) full_recorded_encoder_obs: use recorded sonic.encoder_obs directly

Use this to test whether these three SMPL-mode encoder blocks are the dominant
source of replay mismatch.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


def _load_playback_module(repo_root: Path):
    mod_path = repo_root / "gear_sonic/scripts/playback_lerobot.py"
    spec = importlib.util.spec_from_file_location("playback_lerobot", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare replay with recorded SMPL encoder blocks.")
    p.add_argument("--dataset_dir", required=True, help="LeRobot dataset dir (e.g. ./lerobot_dataset_2.4)")
    p.add_argument("--episode", type=int, default=0, help="Episode index in LeRobot dataset")
    p.add_argument(
        "--recorded_episode_dir",
        required=True,
        help="Original teleop recording dir containing sonic.npz (e.g. recordings/20260410_122727_ep0002)",
    )
    p.add_argument("--encoder_model", required=True, help="Path to model_encoder.onnx")
    p.add_argument("--decoder_model", required=True, help="Path to model_decoder.onnx")
    p.add_argument("--fps", type=float, default=20.0, help="Policy frequency used in replay logic")
    p.add_argument("--num_frames", type=int, default=80, help="How many frames to compare from episode start")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    mod = _load_playback_module(repo_root)

    # Load dataset episode with sonic columns
    (
        states,
        actions,
        _task_indices,
        _base_pos,
        base_quat,
        base_ang_vel_rec,
        body_dq_rec,
        enc_obs_rec,
        _token_rec,
        _dec_obs_rec,
        _dec_action_rec,
        _q_target_cmd_rec,
        smpl_joints,
        body_root_quat,
    ) = mod._load_episode(args.dataset_dir, args.episode, sonic=True)

    # Load original recording model I/O
    sonic_npz = Path(args.recorded_episode_dir) / "sonic.npz"
    if not sonic_npz.exists():
        raise FileNotFoundError(f"Missing sonic.npz at {sonic_npz}")
    sonic_raw = np.load(sonic_npz, allow_pickle=True)
    if "encoder_obs" not in sonic_raw.files:
        raise KeyError(f"{sonic_npz} missing key 'encoder_obs'")
    rec_encoder_obs = sonic_raw["encoder_obs"].astype(np.float64)

    # Build first-frame replay initialization state directly from recorded SONIC
    # measured signals (same 43-DoF layout used by playback_lerobot.py):
    #   [body(0:22), left_hand(7), body(22:29), right_hand(7)]
    required_init_keys = ("body_q_measured", "left_hand_q_measured", "right_hand_q_measured")
    missing_init_keys = [k for k in required_init_keys if k not in sonic_raw.files]
    if missing_init_keys:
        raise KeyError(
            f"{sonic_npz} missing required first-state keys: {missing_init_keys}"
        )
    body_q_measured = sonic_raw["body_q_measured"].astype(np.float64)
    left_hand_q_measured = sonic_raw["left_hand_q_measured"].astype(np.float64)
    right_hand_q_measured = sonic_raw["right_hand_q_measured"].astype(np.float64)
    if body_q_measured.shape[0] == 0:
        raise ValueError(f"{sonic_npz} has empty body_q_measured")
    init_state43_from_recorded = np.zeros((43,), dtype=np.float64)
    init_state43_from_recorded[0:22] = body_q_measured[0, 0:22]
    init_state43_from_recorded[22:29] = left_hand_q_measured[0]
    init_state43_from_recorded[29:36] = body_q_measured[0, 22:29]
    init_state43_from_recorded[36:43] = right_hand_q_measured[0]

    T = min(len(states), len(rec_encoder_obs))
    N = min(args.num_frames, T)
    if N <= 0:
        raise ValueError("No frames to compare.")

    if base_quat is None:
        base_quat = np.tile([1.0, 0.0, 0.0, 0.0], (len(states), 1))

    bq_arr = base_quat
    body_root_quat_corr, wrist_all = mod.SonicRunner.precompute(
        smpl_joints, body_root_quat, bq_arr, actions
    )

    s_smpl, e_smpl = mod._ENC_OBS_LAYOUT["smpl_joints_10frame_step1"]
    s_anchor, e_anchor = mod._ENC_OBS_LAYOUT["smpl_anchor_orientation_10frame_step1"]
    s_wrist, e_wrist = mod._ENC_OBS_LAYOUT["motion_joint_positions_wrists_10frame_step1"]

    def run_variant(name: str, inject_three_blocks: bool, use_full_recorded_encoder_obs: bool):
        runner = mod.SonicRunner(args.encoder_model, args.decoder_model, fps=args.fps, closed_loop=False)
        bq0 = base_quat[0].astype(np.float64)
        runner.reset_history(
            init_state43_from_recorded,
            bq0,
            actions[0],
            body29_vel0_mujoco=body_dq_rec[0] if body_dq_rec is not None else None,
            base_ang_vel0=base_ang_vel_rec[0] if base_ang_vel_rec is not None else None,
        )

        out = []
        for i in range(N):
            bq = base_quat[i].astype(np.float64)
            state43_i = init_state43_from_recorded if i == 0 else states[i]

            encoder_override = None
            if use_full_recorded_encoder_obs:
                encoder_override = rec_encoder_obs[i].astype(np.float32, copy=False)
            elif inject_three_blocks:
                # Build default replay encoder obs, then overwrite the 3 SMPL-mode blocks
                enc = np.zeros(mod._ENC_INPUT_DIM, dtype=np.float32)
                ss, ee = mod._ENC_OBS_LAYOUT["encoder_mode_4"]
                enc[ss] = 2.0

                smpl_win = mod.SonicRunner.future_window(smpl_joints, i)
                brq_win = mod.SonicRunner.future_window(body_root_quat_corr, i)
                wrist_win = mod.SonicRunner.future_window(wrist_all, i)

                enc[s_smpl:e_smpl] = smpl_win.reshape(-1)
                anchor_ori_win = np.array(
                    [mod._smpl_anchor_ori_6d(bq, brq_win[fi]) for fi in range(mod._HISTORY_LEN)],
                    dtype=np.float32,
                )
                enc[s_anchor:e_anchor] = anchor_ori_win.reshape(-1)
                enc[s_wrist:e_wrist] = wrist_win.reshape(-1)

                rec = rec_encoder_obs[i]
                enc[s_smpl:e_smpl] = rec[s_smpl:e_smpl]
                enc[s_anchor:e_anchor] = rec[s_anchor:e_anchor]
                enc[s_wrist:e_wrist] = rec[s_wrist:e_wrist]
                encoder_override = enc

            sonic_state43 = runner.step(
                state43=state43_i,
                action43=actions[i],
                base_quat=bq,
                smpl_joints_win=mod.SonicRunner.future_window(smpl_joints, i),
                body_root_quat_win=mod.SonicRunner.future_window(body_root_quat_corr, i),
                wrist_win=mod.SonicRunner.future_window(wrist_all, i),
                body29_vel_mujoco=body_dq_rec[i] if body_dq_rec is not None else None,
                base_ang_vel=base_ang_vel_rec[i] if base_ang_vel_rec is not None else None,
                encoder_obs_rec=encoder_override,
            )
            out.append(sonic_state43)

        out = np.stack(out)
        pred_body29 = out[:, mod._BODY_IDX]
        ref_action_body29 = actions[:N][:, mod._BODY_IDX]
        ref_state_body29 = states[:N][:, mod._BODY_IDX]
        l2_action = np.linalg.norm(pred_body29 - ref_action_body29, axis=1)
        l2_state = np.linalg.norm(pred_body29 - ref_state_body29, axis=1)

        print(f"\n[{name}]")
        print(
            "  vs action.body29:"
            f" mean={l2_action.mean():.6f}, max={l2_action.max():.6f}, frame0={l2_action[0]:.6f}"
        )
        print(
            "  vs state.body29 :"
            f" mean={l2_state.mean():.6f}, max={l2_state.max():.6f}, frame0={l2_state[0]:.6f}"
        )

    print(f"Comparing first N={N} frames")
    print(f"dataset_dir={args.dataset_dir}, recorded_episode_dir={args.recorded_episode_dir}")
    init_delta = float(np.linalg.norm(init_state43_from_recorded - states[0].astype(np.float64)))
    print(f"first-state init: recorded sonic measured frame0 (L2 vs dataset state[0] = {init_delta:.6f})")

    run_variant(
        name="baseline_rebuild",
        inject_three_blocks=False,
        use_full_recorded_encoder_obs=False,
    )
    run_variant(
        name="inject_recorded_smpl_blocks",
        inject_three_blocks=True,
        use_full_recorded_encoder_obs=False,
    )
    run_variant(
        name="full_recorded_encoder_obs",
        inject_three_blocks=False,
        use_full_recorded_encoder_obs=True,
    )


if __name__ == "__main__":
    main()
