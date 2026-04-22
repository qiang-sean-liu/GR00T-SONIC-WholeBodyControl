"""Replay LeRobot episode using recorded SMPL/decoder history inputs.

This wrapper keeps `gear_sonic/scripts/playback_lerobot.py` unchanged and
injects a custom loading behavior:

- Use recorded `sonic.encoder_obs` only for SMPL-mode encoder slices:
    * smpl_joints_10frame_step1
    * smpl_anchor_orientation_10frame_step1
    * motion_joint_positions_wrists_10frame_step1
  (+ mode id set to 2)
- Use recorded `sonic.decoder_obs` directly for decoder input (so 10-frame
  history terms are not reconstructed by replay script).
- Recompute decoder output from that recorded decoder input (do NOT shortcut
  with recorded `decoder_action_raw` / `q_target_cmd`).

Use this script when you want to isolate mismatch caused by reconstructing
encoder/decoder history-related inputs.
"""

from __future__ import annotations

import argparse
import importlib.util
from collections import deque
from pathlib import Path

import numpy as np


def _format_first_big(err: np.ndarray, threshold: float) -> str:
    idx = np.where(err > threshold)[0]
    return "none" if len(idx) == 0 else str(int(idx[0]))


def _import_playback_module(repo_root: Path):
    mod_path = repo_root / "gear_sonic/scripts/playback_lerobot.py"
    spec = importlib.util.spec_from_file_location("playback_lerobot", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay using recorded encoder SMPL blocks and recorded decoder_obs (wrapper around playback_lerobot.py)."
    )
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--env_name", required=True)
    p.add_argument("--output_video", default=None)
    p.add_argument("--camera", default=None, help="Single camera (legacy shorthand)")
    p.add_argument("--cameras", nargs="+", default=None, help="Multiple cameras")
    p.add_argument("--no_viewer", action="store_true")
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--video_width", type=int, default=640)
    p.add_argument("--video_height", type=int, default=360)
    p.add_argument("--sonic_encoder", required=True, help="Path to model_encoder.onnx")
    p.add_argument("--sonic_decoder", required=True, help="Path to model_decoder.onnx")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--physics", action="store_true")
    p.add_argument("--upper_body_from_action", action="store_true")
    p.add_argument(
        "--force_recorded_prefix_frames",
        type=int,
        default=11,
        help=(
            "Number of initial frames to force from recorded data for state/encoder "
            "alignment in policy replay."
        ),
    )
    p.add_argument(
        "--force_frame0_recorded_state",
        action="store_true",
        help=(
            "In SONIC mode, force frame-0 displayed output to recorded state43 "
            "while still running SONIC step internally."
        ),
    )
    p.add_argument(
        "--recorded_episode_dir",
        default=None,
        help="Optional raw recording dir containing sonic.npz; if set, use its frame-0 measured state to seed replay state[0].",
    )
    p.add_argument(
        "--indicator_big_err_encoder",
        type=float,
        default=1e-3,
        help="Threshold for flagging a large per-frame encoder token L2 error.",
    )
    p.add_argument(
        "--indicator_big_err_decoder",
        type=float,
        default=1e-3,
        help="Threshold for flagging a large per-frame decoder action L2 error.",
    )
    p.add_argument(
        "--indicator_report_txt",
        default=None,
        help="Write per-frame encoder/decoder I/O comparison report to this txt path.",
    )
    return p.parse_args()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pb = _import_playback_module(repo_root)
    args = _parse_args()
    recorded_prefix_frames = max(0, int(args.force_recorded_prefix_frames))
    token_prefix_frames: np.ndarray | None = None
    decoder_obs_prefix_frames: np.ndarray | None = None
    decoder_action_prefix_frames: np.ndarray | None = None
    q_target_prefix_frames: np.ndarray | None = None
    init_state43_from_recorded: np.ndarray | None = None
    recorded_state43_from_raw: np.ndarray | None = None
    if args.recorded_episode_dir:
        sonic_npz = Path(args.recorded_episode_dir).expanduser().resolve() / "sonic.npz"
        if not sonic_npz.exists():
            raise FileNotFoundError(f"Missing sonic.npz at {sonic_npz}")
        sonic_raw = np.load(sonic_npz, allow_pickle=True)
        required_keys = ("body_q_measured", "left_hand_q_measured", "right_hand_q_measured")
        missing_keys = [k for k in required_keys if k not in sonic_raw.files]
        if missing_keys:
            raise KeyError(f"{sonic_npz} missing required keys for state init: {missing_keys}")
        body_q_measured = sonic_raw["body_q_measured"].astype(np.float64)
        left_hand_q_measured = sonic_raw["left_hand_q_measured"].astype(np.float64)
        right_hand_q_measured = sonic_raw["right_hand_q_measured"].astype(np.float64)
        if len(body_q_measured) == 0:
            raise ValueError(f"{sonic_npz} has empty body_q_measured")
        recorded_state43_from_raw = np.zeros((len(body_q_measured), 43), dtype=np.float64)
        recorded_state43_from_raw[:, 0:22] = body_q_measured[:, 0:22]
        recorded_state43_from_raw[:, 22:29] = left_hand_q_measured
        recorded_state43_from_raw[:, 29:36] = body_q_measured[:, 22:29]
        recorded_state43_from_raw[:, 36:43] = right_hand_q_measured
        init_state43_from_recorded = np.zeros((43,), dtype=np.float64)
        init_state43_from_recorded[0:22] = body_q_measured[0, 0:22]
        init_state43_from_recorded[22:29] = left_hand_q_measured[0]
        init_state43_from_recorded[29:36] = body_q_measured[0, 22:29]
        init_state43_from_recorded[36:43] = right_hand_q_measured[0]

    # Save original loader and replace with a wrapper that:
    # 1) keeps only recorded SMPL-mode encoder slices from sonic.encoder_obs
    # 2) uses recorded decoder_obs directly (history terms from recording)
    # 3) disables direct replay of recorded decoder_action/q_target outputs
    original_load_episode = pb._load_episode

    def _load_episode_recorded_smpl_blocks(dataset_dir: str, episode: int, sonic: bool = False):
        (
            states,
            actions,
            task_indices,
            base_pos,
            base_quat,
            base_ang_vel,
            body_dq,
            enc_obs_rec,
            token_rec,
            dec_obs_rec,
            dec_action_rec,
            q_target_cmd,
            smpl_joints,
            body_root_quat,
        ) = original_load_episode(dataset_dir, episode, sonic=sonic)

        if recorded_state43_from_raw is not None and len(states) > 0:
            states = states.copy()
            k_state = min(recorded_prefix_frames, len(states), len(recorded_state43_from_raw))
            old_state0 = states[0].astype(np.float64, copy=False)
            if k_state > 0:
                states[:k_state] = recorded_state43_from_raw[:k_state].astype(states.dtype, copy=False)
            init_l2 = float(
                np.linalg.norm(states[0].astype(np.float64) - old_state0)
            )
            print(
                "Replay state prefix init: using recorded sonic measured data "
                f"for first {k_state} frame(s) (frame0 L2 vs dataset state[0] = {init_l2:.6f})"
            )

        if sonic:
            if enc_obs_rec is None:
                raise ValueError(
                    "Dataset does not contain sonic.encoder_obs. "
                    "Need model-IO recording to use recorded SMPL blocks."
                )
            if dec_obs_rec is None:
                raise ValueError(
                    "Dataset does not contain sonic.decoder_obs. "
                    "Need model-IO recording to use recorded decoder history input."
                )
            enc_obs_smpl_only = np.zeros_like(enc_obs_rec, dtype=np.float64)
            # mode_id = 2 for SMPL
            enc_obs_smpl_only[:, 0] = 2.0

            s, e = pb._ENC_OBS_LAYOUT["smpl_joints_10frame_step1"]
            enc_obs_smpl_only[:, s:e] = enc_obs_rec[:, s:e]
            s, e = pb._ENC_OBS_LAYOUT["smpl_anchor_orientation_10frame_step1"]
            enc_obs_smpl_only[:, s:e] = enc_obs_rec[:, s:e]
            s, e = pb._ENC_OBS_LAYOUT["motion_joint_positions_wrists_10frame_step1"]
            enc_obs_smpl_only[:, s:e] = enc_obs_rec[:, s:e]
            k_enc = min(recorded_prefix_frames, len(enc_obs_rec))
            if k_enc > 0:
                # For initial frames, use full recorded encoder_obs so encoder inputs
                # are exactly aligned before switching to SMPL-only injection mode.
                enc_obs_smpl_only[:k_enc] = enc_obs_rec[:k_enc]

            # For model outputs, keep a fully recorded prefix when available.
            if token_rec is not None:
                k_tok = min(recorded_prefix_frames, len(token_rec))
                if k_tok > 0:
                    token_prefix_frames = token_rec[:k_tok].astype(np.float32, copy=True)
            else:
                k_tok = 0

            # For decoder inputs, do the same prefix strategy:
            # first K frames use recorded decoder_obs, then replay reconstructs decoder obs.
            k_dec = min(recorded_prefix_frames, len(dec_obs_rec))
            if k_dec > 0:
                decoder_obs_prefix_frames = dec_obs_rec[:k_dec].astype(np.float32, copy=True)
            else:
                decoder_obs_prefix_frames = None

            if dec_action_rec is not None:
                k_draw = min(recorded_prefix_frames, len(dec_action_rec))
                if k_draw > 0:
                    decoder_action_prefix_frames = dec_action_rec[:k_draw].astype(np.float32, copy=True)
            else:
                k_draw = 0

            if q_target_cmd is not None:
                k_qcmd = min(recorded_prefix_frames, len(q_target_cmd))
                if k_qcmd > 0:
                    q_target_prefix_frames = q_target_cmd[:k_qcmd].astype(np.float64, copy=True)
            else:
                k_qcmd = 0

            # Decoder input comes directly from recorded dec_obs (history terms).
            # Disable output shortcuts so decoder output is recomputed.
            token_rec = None
            dec_action_rec = None
            q_target_cmd = None
            enc_obs_rec = enc_obs_smpl_only
            dec_obs_rec = None

            print(
                "Model I/O mode: recorded SMPL encoder blocks + recorded decoder_obs "
                "(decoder_action/q_target recomputed); "
                f"first {k_enc} frame(s) use full recorded encoder_obs, "
                f"first {k_dec} frame(s) use recorded decoder_obs, "
                f"first {k_tok} frame(s) force token_state, "
                f"first {k_draw} frame(s) force decoder_action_raw, "
                f"first {k_qcmd} frame(s) force q_target_cmd"
            )

        return (
            states,
            actions,
            task_indices,
            base_pos,
            base_quat,
            base_ang_vel,
            body_dq,
            enc_obs_rec,
            token_rec,
            dec_obs_rec,
            dec_action_rec,
            q_target_cmd,
            smpl_joints,
            body_root_quat,
        )

    setattr(pb, "_load_episode", _load_episode_recorded_smpl_blocks)

    if args.force_frame0_recorded_state:
        # Also force the viewer's pre-loop initial pose, so the simulator window
        # does not briefly show the default standing posture before frame 1.
        (
            states_init,
            _actions_init,
            _task_indices_init,
            base_pos_init,
            base_quat_init,
            _base_ang_vel_init,
            _body_dq_init,
            _enc_obs_init,
            _token_init,
            _dec_obs_init,
            _dec_action_init,
            _q_target_init,
            _smpl_init,
            _body_root_init,
        ) = pb._load_episode(args.dataset_dir, args.episode, sonic=True)
        preloop_state0 = states_init[0] if len(states_init) > 0 else None
        preloop_bp0 = base_pos_init[0] if base_pos_init is not None and len(base_pos_init) > 0 else None
        preloop_bq0 = base_quat_init[0] if base_quat_init is not None and len(base_quat_init) > 0 else None

        original_launch_passive = pb.mujoco.viewer.launch_passive

        def _launch_passive_with_frame0(model, data, *lp_args, **lp_kwargs):
            if preloop_state0 is not None:
                body_jids, left_jids, right_jids = pb._build_joint_indices(model)
                root_jid_list = [
                    i for i in range(model.njnt) if model.joint(i).type == pb.mujoco.mjtJoint.mjJNT_FREE
                ]
                root_jid = root_jid_list[0] if root_jid_list else None
                pb._set_qpos(
                    data,
                    model,
                    body_jids,
                    left_jids,
                    right_jids,
                    preloop_state0,
                    root_jid=root_jid,
                    base_pos=preloop_bp0,
                    base_quat=preloop_bq0,
                )
                pb.mujoco.mj_forward(model, data)
            return original_launch_passive(model, data, *lp_args, **lp_kwargs)

        pb.mujoco.viewer.launch_passive = _launch_passive_with_frame0

    if args.cameras:
        cameras = args.cameras
    elif args.camera:
        cameras = [args.camera]
    else:
        cameras = ["overview"]

    sonic_runner = pb.SonicRunner(
        args.sonic_encoder,
        args.sonic_decoder,
        fps=args.fps,
        closed_loop=False,
    )
    env_state_for_next: dict[str, np.ndarray | None] | None = None
    transition_refs: dict[str, np.ndarray | None] = {
        "states": None,
        "base_quat": None,
        "base_ang_vel": None,
        "body_dq": None,
        "decoder_action_raw": None,
    }

    def _seed_decoder_history_from_forced_prefix(runner: object, prefix_frames: int) -> None:
        """Reseed runner history right after teacher-forcing window.

        This makes the first free-running step use history consistent with
        the previous forced frames.
        """
        if prefix_frames <= 0:
            return
        states_ref = transition_refs["states"]
        base_quat_ref = transition_refs["base_quat"]
        if states_ref is None or base_quat_ref is None:
            return

        hist_len = pb._HISTORY_LEN
        # We seed one-step earlier than the handoff window because SonicRunner.step
        # appends the current frame before building decoder_obs. This ensures:
        # frame (prefix+1) uses history from frames [2..prefix] in 1-based indexing.
        start = max(0, (prefix_frames - 1) - hist_len)
        idxs = list(range(start, max(0, prefix_frames - 1)))
        if not idxs:
            return

        base_ang_ref = transition_refs["base_ang_vel"]
        body_dq_ref = transition_refs["body_dq"]
        dec_action_ref = transition_refs["decoder_action_raw"]

        pos_hist: list[np.ndarray] = []
        vel_hist: list[np.ndarray] = []
        ang_hist: list[np.ndarray] = []
        grav_hist: list[np.ndarray] = []
        act_hist: list[np.ndarray] = []

        for i in idxs:
            s43 = states_ref[i].astype(np.float64)
            body29_muj = np.concatenate([s43[0:22], s43[29:36]])
            body29_dev = body29_muj - pb._DEFAULT_ANGLES
            body29_il = np.zeros(29, dtype=np.float32)
            body29_il[pb._ISAACLAB_TO_MUJOCO] = body29_dev.astype(np.float32)
            pos_hist.append(body29_il)

            if body_dq_ref is not None:
                body29_vel_il = np.zeros(29, dtype=np.float32)
                body29_vel_il[pb._ISAACLAB_TO_MUJOCO] = body_dq_ref[i].astype(np.float32)
            else:
                body29_vel_il = np.zeros(29, dtype=np.float32)
            vel_hist.append(body29_vel_il)

            if base_ang_ref is not None:
                ang_hist.append(base_ang_ref[i].astype(np.float32))
            else:
                ang_hist.append(np.zeros(3, dtype=np.float32))

            grav_hist.append(pb._gravity_dir_body(base_quat_ref[i].astype(np.float64)).astype(np.float32))

            if dec_action_ref is not None and i < len(dec_action_ref):
                act_hist.append(dec_action_ref[i].astype(np.float32))
            else:
                act_hist.append(np.zeros(29, dtype=np.float32))

        while len(pos_hist) < hist_len:
            pos_hist.insert(0, pos_hist[0].copy())
            vel_hist.insert(0, vel_hist[0].copy())
            ang_hist.insert(0, ang_hist[0].copy())
            grav_hist.insert(0, grav_hist[0].copy())
            act_hist.insert(0, act_hist[0].copy())

        pos_hist = pos_hist[-hist_len:]
        vel_hist = vel_hist[-hist_len:]
        ang_hist = ang_hist[-hist_len:]
        grav_hist = grav_hist[-hist_len:]
        act_hist = act_hist[-hist_len:]

        runner._joint_pos_hist = deque(pos_hist, maxlen=hist_len)
        runner._joint_vel_hist = deque(vel_hist, maxlen=hist_len)
        runner._ang_vel_hist = deque(ang_hist, maxlen=hist_len)
        runner._gravity_hist = deque(grav_hist, maxlen=hist_len)
        runner._last_action_hist = deque(act_hist, maxlen=hist_len)
        runner._prev_base_quat = base_quat_ref[prefix_frames - 1].astype(np.float64).copy()
        runner._prev_joint_pos_il = pos_hist[-1].copy()
        runner._prev_policy_body29_il = pos_hist[-1].copy()
    if (
        token_prefix_frames is not None
        or decoder_obs_prefix_frames is not None
        or decoder_action_prefix_frames is not None
        or q_target_prefix_frames is not None
    ):
        original_step_with_model_prefix = sonic_runner.step
        step_counter_model = {"i": 0}

        def _step_with_model_prefix(*step_args, **step_kwargs):
            i = step_counter_model["i"]
            if token_prefix_frames is not None and i < len(token_prefix_frames):
                step_kwargs["token_rec"] = token_prefix_frames[i]
            if decoder_obs_prefix_frames is not None and i < len(decoder_obs_prefix_frames):
                step_kwargs["decoder_obs_rec"] = decoder_obs_prefix_frames[i]
            if decoder_action_prefix_frames is not None and i < len(decoder_action_prefix_frames):
                step_kwargs["decoder_action_raw_rec"] = decoder_action_prefix_frames[i]
            if q_target_prefix_frames is not None and i < len(q_target_prefix_frames):
                step_kwargs["q_target_cmd_rec"] = q_target_prefix_frames[i]
            out = original_step_with_model_prefix(*step_args, **step_kwargs)
            step_counter_model["i"] += 1
            return out

        sonic_runner.step = _step_with_model_prefix  # type: ignore[method-assign]
        print(
            "Model I/O prefix override active "
            f"(token={0 if token_prefix_frames is None else len(token_prefix_frames)}, "
            f"dec_obs={0 if decoder_obs_prefix_frames is None else len(decoder_obs_prefix_frames)}, "
            f"dec_action={0 if decoder_action_prefix_frames is None else len(decoder_action_prefix_frames)}, "
            f"q_target={0 if q_target_prefix_frames is None else len(q_target_prefix_frames)})"
        )

    if decoder_obs_prefix_frames is not None:
        original_step_with_dec_prefix = sonic_runner.step
        step_counter_dec = {"i": 0}

        def _step_with_decoder_prefix(*step_args, **step_kwargs):
            i = step_counter_dec["i"]
            if i < len(decoder_obs_prefix_frames):
                step_kwargs["decoder_obs_rec"] = decoder_obs_prefix_frames[i]
            out = original_step_with_dec_prefix(*step_args, **step_kwargs)
            step_counter_dec["i"] += 1
            return out

        sonic_runner.step = _step_with_decoder_prefix  # type: ignore[method-assign]
        print(
            "Decoder input override: use recorded decoder_obs "
            f"for first {len(decoder_obs_prefix_frames)} frame(s)"
        )

    if not args.physics and recorded_prefix_frames > 0:
        # From frame (prefix+1) onward, use the replay environment observation
        # (previous displayed SONIC output) as state input instead of dataset state.
        original_step_with_env_obs = sonic_runner.step
        step_counter_env = {"i": 0}
        env_state_for_next = {"value": None}
        hist_len = pb._HISTORY_LEN
        hist_pos: deque[np.ndarray] = deque(maxlen=hist_len)
        hist_vel: deque[np.ndarray] = deque(maxlen=hist_len)
        hist_ang: deque[np.ndarray] = deque(maxlen=hist_len)
        hist_grav: deque[np.ndarray] = deque(maxlen=hist_len)
        hist_act: deque[np.ndarray] = deque(maxlen=hist_len)

        def _body29_il_from_state43(state43_in: np.ndarray) -> np.ndarray:
            body29_muj = np.concatenate([state43_in[0:22], state43_in[29:36]]).astype(np.float64)
            body29_dev = body29_muj - pb._DEFAULT_ANGLES
            body29_il = np.zeros(29, dtype=np.float32)
            body29_il[pb._ISAACLAB_TO_MUJOCO] = body29_dev.astype(np.float32)
            return body29_il

        def _step_with_env_obs_after_prefix(*step_args, **step_kwargs):
            i = step_counter_env["i"]
            if i == recorded_prefix_frames:
                _seed_decoder_history_from_forced_prefix(sonic_runner, recorded_prefix_frames)
            # Bridge step at i == recorded_prefix_frames uses the current frame state.
            # From the NEXT frame onward, feed back environment state.
            if i > recorded_prefix_frames and env_state_for_next["value"] is not None:
                env_state = env_state_for_next["value"]
                if "state43" in step_kwargs:
                    step_kwargs["state43"] = env_state
                elif len(step_args) > 0:
                    step_args = (env_state, *step_args[1:])

            # After teacher-forcing prefix, explicitly feed decoder history buffers
            # from our rolling 10-frame buffer.
            if i >= recorded_prefix_frames and len(hist_pos) == hist_len:
                sonic_runner._joint_pos_hist = deque([v.copy() for v in hist_pos], maxlen=hist_len)
                sonic_runner._joint_vel_hist = deque([v.copy() for v in hist_vel], maxlen=hist_len)
                sonic_runner._ang_vel_hist = deque([v.copy() for v in hist_ang], maxlen=hist_len)
                sonic_runner._gravity_hist = deque([v.copy() for v in hist_grav], maxlen=hist_len)
                sonic_runner._last_action_hist = deque([v.copy() for v in hist_act], maxlen=hist_len)

            # Capture current step inputs for history update.
            state43_cur = step_kwargs.get("state43", step_args[0] if len(step_args) > 0 else None)
            base_quat_cur = step_kwargs.get("base_quat", step_args[2] if len(step_args) > 2 else None)
            body_dq_cur = step_kwargs.get("body29_vel_mujoco", None)
            base_ang_cur = step_kwargs.get("base_ang_vel", None)

            out = original_step_with_env_obs(*step_args, **step_kwargs)
            # If no display-state override is active, next env state follows model output.
            if not args.force_frame0_recorded_state:
                env_state_for_next["value"] = np.array(out, dtype=np.float32, copy=False)

            # Update rolling 10-frame history buffer from this step.
            if state43_cur is not None and base_quat_cur is not None:
                # Use current frame output for history shift correctness.
                state43_arr = np.array(out, dtype=np.float64, copy=False)
                base_quat_arr = np.array(base_quat_cur, dtype=np.float64, copy=False)
                body29_il = _body29_il_from_state43(state43_arr)
                if body_dq_cur is not None:
                    body29_vel_il = np.zeros(29, dtype=np.float32)
                    body29_vel_il[pb._ISAACLAB_TO_MUJOCO] = np.array(body_dq_cur, dtype=np.float32, copy=False)
                elif len(hist_pos) > 0:
                    body29_vel_il = (body29_il - hist_pos[-1]) / float(1.0 / args.fps)
                else:
                    body29_vel_il = np.zeros(29, dtype=np.float32)
                if base_ang_cur is not None:
                    ang_vel = np.array(base_ang_cur, dtype=np.float32, copy=False)
                else:
                    ang_vel = np.zeros(3, dtype=np.float32)
                gravity = pb._gravity_dir_body(base_quat_arr).astype(np.float32)
                if hasattr(sonic_runner, "_last_action_hist") and len(sonic_runner._last_action_hist) > 0:
                    last_action = np.array(sonic_runner._last_action_hist[-1], dtype=np.float32, copy=False)
                else:
                    last_action = np.zeros(29, dtype=np.float32)
                hist_pos.append(body29_il)
                hist_vel.append(body29_vel_il)
                hist_ang.append(ang_vel)
                hist_grav.append(gravity)
                hist_act.append(last_action)
            step_counter_env["i"] += 1
            return out

        sonic_runner.step = _step_with_env_obs_after_prefix  # type: ignore[method-assign]
        print(
            "State input mode: using environment state/obs after prefix "
            f"(from frame index {recorded_prefix_frames})"
        )
        print(
            f"Decoder history buffer: rolling {hist_len} frames "
            "for his_body_joint_positions/velocities/base_ang_vel/gravity/last_actions"
        )

    if args.force_frame0_recorded_state:
        original_step = sonic_runner.step
        step_counter = {"i": 0}

        def _step_force_frame0_recorded_state(*step_args, **step_kwargs):
            out = original_step(*step_args, **step_kwargs)
            if step_counter["i"] < recorded_prefix_frames:
                state43 = step_kwargs.get("state43", None)
                if state43 is None and len(step_args) > 0:
                    state43 = step_args[0]
                if state43 is not None:
                    out = np.array(state43, dtype=np.float64)
            # Keep env-state feedback consistent with what is actually displayed.
            if env_state_for_next is not None:
                env_state_for_next["value"] = np.array(out, dtype=np.float32, copy=False)
            step_counter["i"] += 1
            return out

        sonic_runner.step = _step_force_frame0_recorded_state  # type: ignore[method-assign]
        print(
            "State override: force displayed SONIC output to recorded state43 "
            f"for first {recorded_prefix_frames} frame(s)"
        )

    # Terminal indicator: compare encoder/decoder outputs
    # between this replay mode and recorded LeRobot model-I/O.
    (
        _states_i,
        _actions_i,
        _task_indices_i,
        _base_pos_i,
        _base_quat_i,
        _base_ang_vel_i,
        _body_dq_i,
        enc_obs_full_i,
        token_rec_i,
        dec_obs_rec_i,
        dec_action_rec_i,
        q_target_cmd_rec_i,
        _smpl_joints_i,
        _body_root_quat_i,
    ) = original_load_episode(args.dataset_dir, args.episode, sonic=True)
    transition_refs["states"] = _states_i
    transition_refs["base_quat"] = _base_quat_i
    transition_refs["base_ang_vel"] = _base_ang_vel_i
    transition_refs["body_dq"] = _body_dq_i
    transition_refs["decoder_action_raw"] = dec_action_rec_i

    print("\n[Indicator] Replay vs recorded model-I/O")
    report_lines: list[str] = []
    report_lines.append("# SONIC playback per-frame I/O comparison report")
    report_lines.append(f"dataset_dir={Path(args.dataset_dir).resolve()}")
    report_lines.append(f"episode={args.episode}")
    report_lines.append(f"env_name={args.env_name}")
    report_lines.append(f"fps={args.fps}")
    report_lines.append(
        f"threshold_encoder={args.indicator_big_err_encoder:.9e},threshold_decoder={args.indicator_big_err_decoder:.9e}"
    )

    if enc_obs_full_i is not None and token_rec_i is not None and len(enc_obs_full_i) > 0:
        n = min(len(enc_obs_full_i), len(token_rec_i))
        token_err = np.zeros((n,), dtype=np.float64)
        enc_in_err_full = np.zeros((n,), dtype=np.float64)
        enc_in_err_smpl = np.zeros((n,), dtype=np.float64)
        token_max_abs = np.zeros((n,), dtype=np.float64)
        for i in range(n):
            enc = np.zeros_like(enc_obs_full_i[i], dtype=np.float32)
            enc[0] = 2.0
            s, e = pb._ENC_OBS_LAYOUT["smpl_joints_10frame_step1"]
            enc[s:e] = enc_obs_full_i[i, s:e].astype(np.float32)
            s, e = pb._ENC_OBS_LAYOUT["smpl_anchor_orientation_10frame_step1"]
            enc[s:e] = enc_obs_full_i[i, s:e].astype(np.float32)
            s, e = pb._ENC_OBS_LAYOUT["motion_joint_positions_wrists_10frame_step1"]
            enc[s:e] = enc_obs_full_i[i, s:e].astype(np.float32)
            token_pred = sonic_runner._enc.run(None, {"obs_dict": enc[np.newaxis]})[0][0]
            token_ref = token_rec_i[i].astype(np.float32)
            token_diff = token_pred - token_ref
            token_err[i] = float(np.linalg.norm(token_diff))
            token_max_abs[i] = float(np.max(np.abs(token_diff)))

            enc_ref = enc_obs_full_i[i].astype(np.float32)
            enc_in_err_full[i] = float(np.linalg.norm(enc - enc_ref))
            smpl_ref = np.concatenate(
                [
                    enc_ref[pb._ENC_OBS_LAYOUT["smpl_joints_10frame_step1"][0] : pb._ENC_OBS_LAYOUT["smpl_joints_10frame_step1"][1]],
                    enc_ref[
                        pb._ENC_OBS_LAYOUT["smpl_anchor_orientation_10frame_step1"][0] : pb._ENC_OBS_LAYOUT[
                            "smpl_anchor_orientation_10frame_step1"
                        ][1]
                    ],
                    enc_ref[
                        pb._ENC_OBS_LAYOUT["motion_joint_positions_wrists_10frame_step1"][0] : pb._ENC_OBS_LAYOUT[
                            "motion_joint_positions_wrists_10frame_step1"
                        ][1]
                    ],
                ]
            )
            smpl_in = np.concatenate(
                [
                    enc[pb._ENC_OBS_LAYOUT["smpl_joints_10frame_step1"][0] : pb._ENC_OBS_LAYOUT["smpl_joints_10frame_step1"][1]],
                    enc[
                        pb._ENC_OBS_LAYOUT["smpl_anchor_orientation_10frame_step1"][0] : pb._ENC_OBS_LAYOUT[
                            "smpl_anchor_orientation_10frame_step1"
                        ][1]
                    ],
                    enc[
                        pb._ENC_OBS_LAYOUT["motion_joint_positions_wrists_10frame_step1"][0] : pb._ENC_OBS_LAYOUT[
                            "motion_joint_positions_wrists_10frame_step1"
                        ][1]
                    ],
                ]
            )
            enc_in_err_smpl[i] = float(np.linalg.norm(smpl_in - smpl_ref))

        token_frame0 = token_err[0]
        token_mean = float(token_err.mean())
        token_max = float(token_err.max())
        token_first_big_txt = _format_first_big(token_err, args.indicator_big_err_encoder)
        print(
            "  encoder token_state:"
            f" frame0_L2={token_frame0:.9e}, mean_L2={token_mean:.9e}, max_L2={token_max:.9e}"
        )
        print(
            "  encoder first big error frame:"
            f" {token_first_big_txt} (threshold={args.indicator_big_err_encoder:.3e})"
        )

        report_lines.append(
            "encoder_summary,"
            f"n={n},"
            f"frame0_token_l2={token_frame0:.9e},mean_token_l2={token_mean:.9e},max_token_l2={token_max:.9e},"
            f"first_big_frame={token_first_big_txt},"
            f"mean_encoder_input_l2_full={float(enc_in_err_full.mean()):.9e},"
            f"mean_encoder_input_l2_smpl={float(enc_in_err_smpl.mean()):.9e}"
        )
        report_lines.append(
            "frame,enc_input_l2_full,enc_input_l2_smpl,enc_output_token_l2,enc_output_token_max_abs"
        )
        for i in range(n):
            report_lines.append(
                f"{i},{enc_in_err_full[i]:.9e},{enc_in_err_smpl[i]:.9e},{token_err[i]:.9e},{token_max_abs[i]:.9e}"
            )
    else:
        print("  encoder token_state: unavailable (missing sonic.encoder_obs/token_state)")
        report_lines.append("encoder_summary,unavailable")

    if dec_obs_rec_i is not None and dec_action_rec_i is not None and len(dec_obs_rec_i) > 0:
        n = min(len(dec_obs_rec_i), len(dec_action_rec_i))
        dec_err = np.zeros((n,), dtype=np.float64)
        dec_in_err = np.zeros((n,), dtype=np.float64)
        dec_max_abs = np.zeros((n,), dtype=np.float64)
        dec_pred_all: list[np.ndarray] = []
        dec_ref_all: list[np.ndarray] = []
        qcmd_pred_all: list[np.ndarray] = []
        qcmd_ref_all: list[np.ndarray] = []
        qcmd_err = np.zeros((n,), dtype=np.float64)
        qcmd_max_abs = np.zeros((n,), dtype=np.float64)
        has_qcmd_ref = q_target_cmd_rec_i is not None and len(q_target_cmd_rec_i) > 0
        if has_qcmd_ref:
            n_qcmd = min(n, len(q_target_cmd_rec_i))
        else:
            n_qcmd = 0
        for i in range(n):
            dec_in = dec_obs_rec_i[i].astype(np.float32, copy=False)
            dec_pred = sonic_runner._dec.run(None, {"obs_dict": dec_in[np.newaxis]})[0][0]
            dec_ref = dec_action_rec_i[i].astype(np.float32)
            dec_diff = dec_pred - dec_ref
            dec_err[i] = float(np.linalg.norm(dec_diff))
            dec_max_abs[i] = float(np.max(np.abs(dec_diff)))
            dec_in_err[i] = 0.0
            dec_pred_all.append(dec_pred.astype(np.float32, copy=False))
            dec_ref_all.append(dec_ref.astype(np.float32, copy=False))
            qcmd_pred = pb._DEFAULT_ANGLES + dec_pred[pb._ISAACLAB_TO_MUJOCO] * pb._ACTION_SCALE
            qcmd_pred_all.append(qcmd_pred.astype(np.float64, copy=False))
            if i < n_qcmd:
                qcmd_ref = q_target_cmd_rec_i[i].astype(np.float32)
                qcmd_diff = qcmd_pred.astype(np.float32) - qcmd_ref
                qcmd_err[i] = float(np.linalg.norm(qcmd_diff))
                qcmd_max_abs[i] = float(np.max(np.abs(qcmd_diff)))
                qcmd_ref_all.append(qcmd_ref.astype(np.float64, copy=False))
            else:
                qcmd_ref_all.append(np.array([], dtype=np.float64))

        dec_frame0 = dec_err[0]
        dec_mean = float(dec_err.mean())
        dec_max = float(dec_err.max())
        dec_first_big_txt = _format_first_big(dec_err, args.indicator_big_err_decoder)
        print(
            "  decoder action_raw:"
            f" frame0_L2={dec_frame0:.9e}, mean_L2={dec_mean:.9e}, max_L2={dec_max:.9e}"
        )
        print(
            "  decoder first big error frame:"
            f" {dec_first_big_txt} (threshold={args.indicator_big_err_decoder:.3e})"
        )
        if has_qcmd_ref:
            qcmd_frame0 = qcmd_err[0]
            qcmd_mean = float(qcmd_err[:n_qcmd].mean())
            qcmd_max = float(qcmd_err[:n_qcmd].max())
            qcmd_first_big_txt = _format_first_big(
                qcmd_err[:n_qcmd], args.indicator_big_err_decoder
            )
            print(
                "  q_target_cmd:"
                f" frame0_L2={qcmd_frame0:.9e}, mean_L2={qcmd_mean:.9e}, max_L2={qcmd_max:.9e}"
            )
            print(
                "  q_target_cmd first big error frame:"
                f" {qcmd_first_big_txt} (threshold={args.indicator_big_err_decoder:.3e})"
            )
        else:
            qcmd_first_big_txt = "unavailable"

        report_lines.append(
            "decoder_summary,"
            f"n={n},"
            f"frame0_action_l2={dec_frame0:.9e},mean_action_l2={dec_mean:.9e},max_action_l2={dec_max:.9e},"
            f"first_big_frame={dec_first_big_txt},"
            f"mean_decoder_input_l2={float(dec_in_err.mean()):.9e},"
            + (
                f"frame0_q_target_cmd_l2={qcmd_err[0]:.9e},"
                f"mean_q_target_cmd_l2={float(qcmd_err[:n_qcmd].mean()):.9e},"
                f"max_q_target_cmd_l2={float(qcmd_err[:n_qcmd].max()):.9e},"
                f"first_big_q_target_cmd_frame={qcmd_first_big_txt}"
                if has_qcmd_ref
                else "q_target_cmd=unavailable"
            )
        )
        report_lines.append(
            "frame,dec_input_l2,dec_output_action_l2,dec_output_action_max_abs,q_target_cmd_l2,q_target_cmd_max_abs"
        )
        for i in range(n):
            if i < n_qcmd:
                qcmd_l2_txt = f"{qcmd_err[i]:.9e}"
                qcmd_max_txt = f"{qcmd_max_abs[i]:.9e}"
            else:
                qcmd_l2_txt = "nan"
                qcmd_max_txt = "nan"
            report_lines.append(
                f"{i},{dec_in_err[i]:.9e},{dec_err[i]:.9e},{dec_max_abs[i]:.9e},{qcmd_l2_txt},{qcmd_max_txt}"
            )

        report_lines.append("decoder_action_raw_vectors_begin")
        report_lines.append(
            "frame|decoder_action_raw_replay|sonic.decoder_action_raw_recorded"
        )
        for i in range(n):
            pred_txt = np.array2string(dec_pred_all[i], precision=9, separator=" ", max_line_width=1_000_000)
            ref_txt = np.array2string(dec_ref_all[i], precision=9, separator=" ", max_line_width=1_000_000)
            report_lines.append(f"{i}|{pred_txt}|{ref_txt}")
        report_lines.append("decoder_action_raw_vectors_end")

        report_lines.append("q_target_cmd_vectors_begin")
        report_lines.append(
            "frame|q_target_cmd_replay|sonic.q_target_cmd_recorded|q_target_cmd_l2"
        )
        for i in range(n):
            pred_txt = np.array2string(qcmd_pred_all[i], precision=9, separator=" ", max_line_width=1_000_000)
            if i < n_qcmd and qcmd_ref_all[i].size > 0:
                ref_txt = np.array2string(qcmd_ref_all[i], precision=9, separator=" ", max_line_width=1_000_000)
                l2_txt = f"{qcmd_err[i]:.9e}"
            else:
                ref_txt = "[]"
                l2_txt = "nan"
            report_lines.append(f"{i}|{pred_txt}|{ref_txt}|{l2_txt}")
        report_lines.append("q_target_cmd_vectors_end")
    else:
        print("  decoder action_raw: unavailable (missing sonic.decoder_obs/decoder_action_raw)")
        report_lines.append("decoder_summary,unavailable")

    if args.indicator_report_txt is None:
        report_path = Path(args.dataset_dir) / f"episode_{args.episode:04d}_model_io_frame_compare.txt"
    else:
        report_path = Path(args.indicator_report_txt)
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"  wrote per-frame report: {report_path}")
    print("[Indicator] end\n")

    pb.playback(
        dataset_dir=args.dataset_dir,
        episode=args.episode,
        env_name=args.env_name,
        output_video=args.output_video,
        cameras=cameras,
        no_viewer=args.no_viewer,
        fps=args.fps,
        video_width=args.video_width,
        video_height=args.video_height,
        sonic_runner=sonic_runner,
        compare=args.compare,
        physics=args.physics,
        upper_body_from_action=args.upper_body_from_action,
    )


if __name__ == "__main__":
    main()
