"""Record SONIC+PICO teleop episodes with explicit START/STOP indicators.

This is a copy-style variant of `record_sonic_teleop.py` that adds clear
operator feedback when recording starts/stops:
  - Audible terminal bell(s)
  - High-visibility text banners in the terminal

Controls remain identical:
  Left grip + A = start / stop episode
  Left grip + B = abort current episode
"""

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Optional

import numpy as np
import zmq

import record_sonic_teleop as base


def _ring_bell(count: int) -> None:
    """Emit audible terminal bell(s) without blocking the recorder loop."""
    canberra = shutil.which("canberra-gtk-play")
    for _ in range(max(0, count)):
        sys.stdout.write("\a")
        sys.stdout.flush()
        if canberra is not None:
            try:
                subprocess.Popen(
                    [canberra, "-i", "bell", "-d", "sonic-recorder"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
        time.sleep(0.06)


def _speak_if_available(text: str, enabled: bool) -> None:
    """Optionally speak status using desktop TTS tools if available."""
    if not enabled:
        return
    speaker = shutil.which("spd-say")
    if speaker is None:
        return
    try:
        subprocess.Popen([speaker, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _banner(title: str, details: str) -> None:
    line = "=" * 72
    print(f"\n{line}")
    print(f"[Recorder] {title}")
    print(f"[Recorder] {details}")
    print(f"{line}\n")


def _desktop_notify(title: str, body: str, enabled: bool) -> None:
    if not enabled:
        return
    notifier = shutil.which("notify-send")
    if notifier is None:
        return
    try:
        subprocess.Popen([notifier, title, body], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _notify_start(episode_idx: int, bells: int, speak: bool, desktop_notify: bool) -> None:
    _ring_bell(bells)
    _banner("### RECORDING STARTED ###", f"Episode {episode_idx} is now recording.")
    _desktop_notify("SONIC Recorder: STARTED", f"Episode {episode_idx} recording.", desktop_notify)
    _speak_if_available(f"Recording started. Episode {episode_idx}.", speak)


def _notify_stop(
    episode_idx: int,
    n_frames: int,
    episode_dir: str,
    bells: int,
    speak: bool,
    desktop_notify: bool,
) -> None:
    _ring_bell(bells)
    _banner(
        "### RECORDING STOPPED ###",
        f"Episode {episode_idx} saved with {n_frames} frames at: {episode_dir}",
    )
    _desktop_notify("SONIC Recorder: STOPPED", f"Episode {episode_idx} saved ({n_frames} frames).", desktop_notify)
    _speak_if_available(f"Recording stopped. Episode {episode_idx} saved.", speak)


def _notify_abort(episode_idx: int, n_frames: int, bells: int, speak: bool, desktop_notify: bool) -> None:
    _ring_bell(bells)
    _banner("### RECORDING ABORTED ###", f"Episode {episode_idx} discarded ({n_frames} buffered frames).")
    _desktop_notify("SONIC Recorder: ABORTED", f"Episode {episode_idx} discarded.", desktop_notify)
    _speak_if_available(f"Recording aborted. Episode {episode_idx} discarded.", speak)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record SONIC+PICO teleop episodes with clear START/STOP indicators.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output_dir", default="./recordings", help="Root directory for saved episodes")
    parser.add_argument("--pose_port", type=int, default=5556, help="ZMQ port for PICO pose stream")
    parser.add_argument("--sonic_port", type=int, default=5557, help="ZMQ port for SONIC g1_debug stream")
    parser.add_argument("--image_port", type=int, default=5555, help="ZMQ port for sim camera images")
    parser.add_argument("--host", default="localhost", help="Host for ZMQ connections")
    parser.add_argument(
        "--base_state_port",
        type=int,
        default=5558,
        help="ZMQ port for ground-truth base state (0 = disabled)",
    )
    parser.add_argument("--no_images", action="store_true", help="Skip camera image recording")
    parser.add_argument("--env_name", default="", help="Scene name saved to meta.json")
    parser.add_argument("--task", default="", help="Task description saved to meta.json")
    parser.add_argument(
        "--start_bells",
        type=int,
        default=2,
        help="Number of terminal bell rings when recording starts (default: 2)",
    )
    parser.add_argument(
        "--stop_bells",
        type=int,
        default=3,
        help="Number of terminal bell rings when recording stops/saves (default: 3)",
    )
    parser.add_argument(
        "--abort_bells",
        type=int,
        default=1,
        help="Number of terminal bell rings when recording is aborted (default: 1)",
    )
    parser.add_argument(
        "--speak_status",
        action="store_true",
        help="Use desktop TTS (spd-say) for start/stop/abort notifications if available",
    )
    parser.add_argument(
        "--desktop_notify",
        action="store_true",
        help="Send desktop notifications (notify-send) on start/stop/abort if available",
    )
    parser.add_argument(
        "--recording_status_hz",
        type=float,
        default=2.0,
        help="How often to refresh REC status line while recording (default: 2.0 Hz)",
    )
    parser.add_argument(
        "--combo_grip_threshold",
        type=float,
        default=0.5,
        help="Grip threshold for fallback A/B combo detection (default: 0.5)",
    )
    parser.add_argument(
        "--combo_cooldown_s",
        type=float,
        default=0.35,
        help="Minimum seconds between fallback combo triggers (default: 0.35)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    stop_event = threading.Event()
    sonic_holder = base._LatestValue()
    image_holder = base._LatestValue()
    base_state_holder = base._LatestValue()

    threading.Thread(
        target=base._sonic_subscriber,
        args=(args.sonic_port, "g1_debug", sonic_holder, stop_event, args.host),
        daemon=True,
        name="sonic-sub",
    ).start()

    if args.base_state_port > 0:
        threading.Thread(
            target=base._base_state_subscriber,
            args=(args.base_state_port, "base_state", base_state_holder, stop_event, args.host),
            daemon=True,
            name="base-state-sub",
        ).start()

    if not args.no_images:
        threading.Thread(
            target=base._image_subscriber,
            args=(args.image_port, image_holder, stop_event, args.host),
            daemon=True,
            name="image-sub",
        ).start()

    ctx = zmq.Context()
    pose_sock = ctx.socket(zmq.SUB)
    pose_sock.setsockopt(zmq.RCVHWM, 10)
    pose_sock.setsockopt(zmq.LINGER, 0)
    pose_sock.setsockopt_string(zmq.SUBSCRIBE, "pose")
    pose_sock.setsockopt(zmq.RCVTIMEO, 500)
    pose_sock.connect(f"tcp://{args.host}:{args.pose_port}")

    print(f"[Recorder] Pose stream: tcp://{args.host}:{args.pose_port}")
    print(f"[Recorder] Output dir:  {os.path.abspath(args.output_dir)}")
    print("[Recorder] Ready.")
    print("[Recorder]   Start/stop : squeeze LEFT grip button (side) + press RIGHT A button")
    print("[Recorder]   Abort      : squeeze LEFT grip button (side) + press RIGHT B button")
    print("[Recorder]   First press = start recording, second press = stop + save")

    recording = False
    buf: Optional[base._EpisodeBuffer] = None
    episode_idx = 0
    pose_msg_count = 0
    last_heartbeat = time.time()
    last_rec_status = 0.0
    showed_rec_line = False
    combo_toggle_last = False
    combo_abort_last = False
    last_toggle_event = 0.0
    last_abort_event = 0.0

    try:
        while True:
            try:
                raw = pose_sock.recv()
            except zmq.Again:
                continue

            pose = base.unpack_pose_message(raw, "pose")
            if pose is None:
                continue

            pose_msg_count += 1

            now = time.time()
            if now - last_heartbeat >= 5.0:
                status = f"RECORDING ({len(buf)} frames)" if recording else "idle"
                sonic_ok = sonic_holder.get() is not None
                base_ok = base_state_holder.get() is not None if args.base_state_port > 0 else None
                base_str = f"  base_state={'ok' if base_ok else 'no data'}" if base_ok is not None else ""
                print(
                    f"[Recorder] pose msgs={pose_msg_count}  status={status}"
                    f"  sonic={'ok' if sonic_ok else 'no data'}{base_str}"
                )
                last_heartbeat = now

            if recording and buf is not None and args.recording_status_hz > 0:
                period = 1.0 / args.recording_status_hz
                if now - last_rec_status >= period:
                    elapsed_s = now - buf.start_wall
                    sys.stdout.write(
                        f"\r[REC ●] episode={episode_idx:04d}  frames={len(buf):6d}  elapsed={elapsed_s:7.2f}s"
                    )
                    sys.stdout.flush()
                    last_rec_status = now
                    showed_rec_line = True
            elif showed_rec_line:
                sys.stdout.write("\n")
                sys.stdout.flush()
                showed_rec_line = False

            toggle = bool(pose.get("toggle_data_collection", np.array([False]))[0])
            abort = bool(pose.get("toggle_data_abort", np.array([False]))[0])

            # Fallback path: derive combo directly from raw button states when present.
            # This avoids missed one-frame toggle pulses under heavy load.
            left_grip = float(pose.get("left_grip", np.array([0.0], dtype=np.float32))[0])
            a_button = bool(pose.get("a_button", np.array([False]))[0])
            b_button = bool(pose.get("b_button", np.array([False]))[0])
            combo_toggle_level = a_button and left_grip > args.combo_grip_threshold
            combo_abort_level = b_button and left_grip > args.combo_grip_threshold

            combo_toggle_edge = (
                combo_toggle_level
                and not combo_toggle_last
                and (now - last_toggle_event) >= args.combo_cooldown_s
            )
            combo_abort_edge = (
                combo_abort_level
                and not combo_abort_last
                and (now - last_abort_event) >= args.combo_cooldown_s
            )
            combo_toggle_last = combo_toggle_level
            combo_abort_last = combo_abort_level

            if combo_toggle_edge:
                toggle = True
                last_toggle_event = now
            if combo_abort_edge:
                abort = True
                last_abort_event = now

            if toggle or abort:
                if showed_rec_line:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    showed_rec_line = False
                print(
                    f"[Recorder] Signal received — toggle={toggle}  abort={abort}"
                    f"  left_grip={left_grip:.2f}"
                    f"  (pose msg #{pose_msg_count})"
                )

            if abort and recording:
                _notify_abort(
                    episode_idx=episode_idx,
                    n_frames=len(buf),
                    bells=args.abort_bells,
                    speak=args.speak_status,
                    desktop_notify=args.desktop_notify,
                )
                recording = False
                buf = None

            if toggle:
                if not recording:
                    recording = True
                    buf = base._EpisodeBuffer()
                    episode_idx += 1
                    _notify_start(
                        episode_idx=episode_idx,
                        bells=args.start_bells,
                        speak=args.speak_status,
                        desktop_notify=args.desktop_notify,
                    )
                else:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    episode_dir = os.path.join(args.output_dir, f"{ts}_ep{episode_idx:04d}")
                    n_frames = len(buf)
                    base.save_episode(buf, episode_dir, env_name=args.env_name, task=args.task)
                    _notify_stop(
                        episode_idx=episode_idx,
                        n_frames=n_frames,
                        episode_dir=episode_dir,
                        bells=args.stop_bells,
                        speak=args.speak_status,
                        desktop_notify=args.desktop_notify,
                    )
                    recording = False
                    buf = None

            if recording and buf is not None:
                img_result = image_holder.get() if not args.no_images else None
                imgs, img_ts = (img_result if img_result is not None else (None, None))

                sonic_frame = sonic_holder.get()
                base_state = base_state_holder.get() if args.base_state_port > 0 else None
                if sonic_frame is not None and base_state is not None:
                    sonic_frame = {**sonic_frame, **base_state}
                elif base_state is not None:
                    sonic_frame = base_state

                buf.append(pico=pose, sonic=sonic_frame, images=imgs, image_ts=img_ts)
                if len(buf) % 100 == 0:
                    print(f"[Recorder]   {len(buf)} frames recorded...")

    except KeyboardInterrupt:
        print("\n[Recorder] Interrupted by user.")
        if recording and buf and len(buf) > 0:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_dir = os.path.join(args.output_dir, f"{ts}_ep{episode_idx:04d}_partial")
            print(f"[Recorder] Saving partial episode ({len(buf)} frames)...")
            base.save_episode(buf, episode_dir, env_name=args.env_name, task=args.task)
    finally:
        if showed_rec_line:
            sys.stdout.write("\n")
            sys.stdout.flush()
        stop_event.set()
        pose_sock.close()
        ctx.term()
        print("[Recorder] Shutdown complete.")


if __name__ == "__main__":
    main()
