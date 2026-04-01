"""Record SONIC+PICO teleop episodes from the gear_sonic pipeline.

Subscribes to three ZMQ streams:
  - Port 5556 (topic 'pose'): PICO human motion, VR data, toggle signals
  - Port 5557 (topic 'g1_debug'): SONIC WBC outputs (target/measured joint positions)
  - Port 5555: sim camera images (requires --enable_image_publish in run_sim_loop.py)

Recording is controlled by the PICO controller:
  Left grip + A = start / stop episode
  Left grip + B = abort (discard current episode)

Output structure per episode::

    <output_dir>/<YYYYMMDD_HHMMSS>_ep<N>/
        pico.npz   -- stacked pose-stream fields (one row per pose tick)
        sonic.npz  -- stacked SONIC output fields (sampled at pose rate)
        images/
            <camera_name>/
                000000.jpg
                000001.jpg
                ...
        meta.json  -- episode metadata (n_frames, duration, timestamps)

Field reference
---------------
pico.npz keys  (see pico_manager_thread_server.py):
  smpl_pose           [T, N, 72]  SMPL body pose
  smpl_joints         [T, N, J, 3] SMPL joint positions
  body_quat_w         [T, N, 4]   body root quaternion (w-first)
  joint_pos           [T, N, 29]  G1 joint positions from motion retargeting
  joint_vel           [T, N, 29]  G1 joint velocities (zeros)
  vr_position         [T, 9]      VR 3-point positions [L-wrist, R-wrist, Neck] × xyz
  vr_orientation      [T, 12]     VR 3-point orientations [L, R, Neck] × wxyz
  frame_index         [T, N]      frame indices
  left/right_trigger  [T, 1]      trigger values
  left/right_grip     [T, 1]      grip values
  left/right_hand_joints [T, 7]   Dex3 hand joint positions
  timestamp_realtime  [T, 1]      wall-clock timestamp (s)
  timestamp_monotonic [T, 1]      monotonic timestamp (s)
  heading_increment   [T, 1]      yaw accumulator change (rad)

sonic.npz keys  (see output_interface.hpp):
  body_q_target       [T, 29]  SONIC target joint positions (MuJoCo order) ← WBC action
  base_trans_target   [T, 3]   target base translation (heading-corrected)
  base_quat_target    [T, 4]   target base quaternion (heading-corrected)
  body_q_measured     [T, 29]  measured joint positions (MuJoCo order)
  base_quat_measured  [T, 4]   measured IMU quaternion
  left/right_hand_q_measured [T, 7]  measured hand joint positions
  vr_3point_position  [T, 9]   VR positions rotated into target body frame
  vr_3point_orientation [T, 12] VR orientations (passed through)
  vr_3point_compliance  [T, 3]  VR compliance values

Usage::

    # Minimal (no images):
    python gear_sonic/scripts/record_sonic_teleop.py --no_images

    # With images (run_sim_loop.py must use --enable_image_publish --enable_offscreen --head_cam):
    python gear_sonic/scripts/record_sonic_teleop.py

    # Custom ports / output directory:
    python gear_sonic/scripts/record_sonic_teleop.py \\
        --output_dir /data/recordings \\
        --pose_port 5556 --sonic_port 5557 --image_port 5555
"""

import argparse
import base64
import json
import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import msgpack
import numpy as np
import zmq

# Must match HEADER_SIZE in zmq_planner_sender.py
_HEADER_SIZE = 1280

_DTYPE_MAP = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
    "bool": np.bool_,
    "u8": np.uint8,
}


# ---------------------------------------------------------------------------
# Wire-format decoders
# ---------------------------------------------------------------------------

def unpack_pose_message(raw: bytes, topic: str = "pose") -> Optional[Dict[str, np.ndarray]]:
    """Decode a Protocol v3 ZMQ pose message into a dict of numpy arrays.

    Wire layout::

        [topic_bytes][_HEADER_SIZE-byte null-padded JSON header][binary payload]

    The JSON header describes each field: name, dtype, shape.
    The binary payload is a concatenation of little-endian arrays in field order.
    """
    topic_bytes = topic.encode("utf-8")
    if not raw.startswith(topic_bytes):
        return None
    offset = len(topic_bytes)
    header_bytes = raw[offset : offset + _HEADER_SIZE]
    payload = raw[offset + _HEADER_SIZE :]

    try:
        header = json.loads(header_bytes.rstrip(b"\x00").decode("utf-8"))
    except json.JSONDecodeError:
        return None

    result: Dict[str, np.ndarray] = {}
    pos = 0
    for field in header.get("fields", []):
        dtype = _DTYPE_MAP.get(field["dtype"], np.float32)
        shape = field["shape"]
        n_elements = int(np.prod(shape)) if shape else 1
        itemsize = np.dtype(dtype).itemsize
        nbytes = n_elements * itemsize
        chunk = payload[pos : pos + nbytes]
        if len(chunk) < nbytes:
            break
        arr = np.frombuffer(chunk, dtype=dtype).copy()
        result[field["name"]] = arr.reshape(shape) if len(shape) > 1 else arr
        pos += nbytes

    return result if result else None


def unpack_sonic_message(raw: bytes, topic: str = "g1_debug") -> Optional[Dict[str, np.ndarray]]:
    """Decode a msgpack message published by deploy.sh on the g1_debug topic.

    Wire layout::

        [topic_bytes][msgpack payload]

    The msgpack payload is a map<string, vector<double>>.
    """
    topic_bytes = topic.encode("utf-8")
    if not raw.startswith(topic_bytes):
        return None
    payload = raw[len(topic_bytes) :]
    try:
        data = msgpack.unpackb(payload)
    except Exception:
        return None
    return {
        (k.decode() if isinstance(k, bytes) else k): np.array(v, dtype=np.float64)
        for k, v in data.items()
    }


def unpack_image_message(raw: bytes) -> Optional[Dict[str, bytes]]:
    """Decode a SensorServer camera image message.

    Returns a dict of camera_name -> raw JPEG bytes (still compressed,
    ready for direct write to .jpg file without re-encoding).
    """
    try:
        data = msgpack.unpackb(raw, raw=False)
        images: Dict[str, bytes] = {}
        for name, encoded in data.get("images", {}).items():
            if isinstance(encoded, str):
                images[name] = base64.b64decode(encoded)
        return images if images else None
    except Exception as e:
        print(f"[Recorder] Image decode error: {e}")
        return None


# ---------------------------------------------------------------------------
# Thread-safe latest-value holder
# ---------------------------------------------------------------------------

class _LatestValue:
    """Holds the most recently received value, thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = None

    def put(self, data) -> None:
        with self._lock:
            self._data = data

    def get(self):
        with self._lock:
            return self._data


# ---------------------------------------------------------------------------
# Background ZMQ subscriber threads
# ---------------------------------------------------------------------------

def _sonic_subscriber(port: int, topic: str, holder: _LatestValue, stop: threading.Event, host: str):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, 10)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.RCVTIMEO, 200)
    sock.connect(f"tcp://{host}:{port}")
    print(f"[Recorder] SONIC subscriber: tcp://{host}:{port} topic='{topic}'")
    while not stop.is_set():
        try:
            raw = sock.recv()
            parsed = unpack_sonic_message(raw, topic)
            if parsed is not None:
                holder.put(parsed)
        except zmq.Again:
            pass
        except Exception as e:
            print(f"[Recorder] SONIC recv error: {e}")
    sock.close()
    ctx.term()


def _image_subscriber(port: int, holder: _LatestValue, stop: threading.Event, host: str):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, 5)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVTIMEO, 200)
    sock.connect(f"tcp://{host}:{port}")
    print(f"[Recorder] Image subscriber: tcp://{host}:{port}")
    while not stop.is_set():
        try:
            raw = sock.recv()
            images = unpack_image_message(raw)
            if images:
                holder.put(images)
        except zmq.Again:
            pass
        except Exception as e:
            print(f"[Recorder] Image recv error: {e}")
    sock.close()
    ctx.term()


# ---------------------------------------------------------------------------
# Episode buffer and saving
# ---------------------------------------------------------------------------

class _EpisodeBuffer:
    def __init__(self):
        self.pico_frames: List[Dict[str, np.ndarray]] = []
        self.sonic_frames: List[Optional[Dict[str, np.ndarray]]] = []
        self.image_frames: List[Optional[Dict[str, bytes]]] = []
        self.start_wall = time.time()

    def append(
        self,
        pico: Dict[str, np.ndarray],
        sonic: Optional[Dict[str, np.ndarray]],
        images: Optional[Dict[str, bytes]],
    ) -> None:
        self.pico_frames.append(pico)
        self.sonic_frames.append(sonic)
        self.image_frames.append(images)

    def __len__(self) -> int:
        return len(self.pico_frames)


def _stack_frames(frames: list) -> Dict[str, np.ndarray]:
    """Stack a list of per-frame dicts into a single dict of stacked arrays."""
    valid = [f for f in frames if f is not None]
    if not valid:
        return {}
    stacked: Dict[str, np.ndarray] = {}
    for key in valid[0]:
        try:
            stacked[key] = np.stack([f[key] for f in valid if key in f], axis=0)
        except Exception:
            pass
    return stacked


def save_episode(buf: _EpisodeBuffer, episode_dir: str) -> None:
    os.makedirs(episode_dir, exist_ok=True)

    # PICO stream
    pico_data = _stack_frames(buf.pico_frames)
    if pico_data:
        np.savez_compressed(os.path.join(episode_dir, "pico.npz"), **pico_data)

    # SONIC stream
    sonic_data = _stack_frames(buf.sonic_frames)
    if sonic_data:
        np.savez_compressed(os.path.join(episode_dir, "sonic.npz"), **sonic_data)

    # Camera images (stored as raw JPEG bytes — no re-encoding)
    n_image_frames = sum(1 for f in buf.image_frames if f is not None)
    if n_image_frames > 0:
        img_root = os.path.join(episode_dir, "images")
        frame_idx = 0
        for frame_imgs in buf.image_frames:
            if frame_imgs is None:
                continue
            for cam_name, jpeg_bytes in frame_imgs.items():
                cam_dir = os.path.join(img_root, cam_name)
                os.makedirs(cam_dir, exist_ok=True)
                with open(os.path.join(cam_dir, f"{frame_idx:06d}.jpg"), "wb") as f:
                    f.write(jpeg_bytes)
            frame_idx += 1

    # Metadata
    duration = time.time() - buf.start_wall
    meta = {
        "n_frames": len(buf),
        "n_image_frames": n_image_frames,
        "duration_s": round(duration, 3),
        "pico_keys": list(pico_data.keys()),
        "sonic_keys": list(sonic_data.keys()),
        "saved_at": datetime.now().isoformat(),
    }
    with open(os.path.join(episode_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[Recorder] Saved: {episode_dir}"
        f"  ({len(buf)} frames, {duration:.1f}s,"
        f" {n_image_frames} image frames)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Record SONIC+PICO teleop episodes (gear_sonic pipeline).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output_dir", default="./recordings",
        help="Root directory for saved episodes (default: ./recordings)",
    )
    parser.add_argument(
        "--pose_port", type=int, default=5556,
        help="ZMQ port for PICO pose stream (default: 5556)",
    )
    parser.add_argument(
        "--sonic_port", type=int, default=5557,
        help="ZMQ port for SONIC g1_debug stream (default: 5557)",
    )
    parser.add_argument(
        "--image_port", type=int, default=5555,
        help="ZMQ port for sim camera images (default: 5555)",
    )
    parser.add_argument(
        "--host", default="localhost",
        help="Host for ZMQ connections (default: localhost)",
    )
    parser.add_argument(
        "--no_images", action="store_true",
        help="Skip camera image recording (faster, smaller output)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    stop_event = threading.Event()
    sonic_holder = _LatestValue()
    image_holder = _LatestValue()

    # Start background subscribers
    threading.Thread(
        target=_sonic_subscriber,
        args=(args.sonic_port, "g1_debug", sonic_holder, stop_event, args.host),
        daemon=True,
        name="sonic-sub",
    ).start()

    if not args.no_images:
        threading.Thread(
            target=_image_subscriber,
            args=(args.image_port, image_holder, stop_event, args.host),
            daemon=True,
            name="image-sub",
        ).start()

    # Main pose stream (also drives the recording state machine)
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
    buf: Optional[_EpisodeBuffer] = None
    episode_idx = 0
    pose_msg_count = 0
    last_heartbeat = time.time()

    try:
        while True:
            try:
                raw = pose_sock.recv()
            except zmq.Again:
                continue

            pose = unpack_pose_message(raw, "pose")
            if pose is None:
                continue

            pose_msg_count += 1

            # Heartbeat: confirm pose stream is flowing (every ~5 s at 50 Hz)
            now = time.time()
            if now - last_heartbeat >= 5.0:
                status = f"RECORDING ({len(buf)} frames)" if recording else "idle"
                sonic_ok = sonic_holder.get() is not None
                print(
                    f"[Recorder] pose msgs={pose_msg_count}  status={status}"
                    f"  sonic={'ok' if sonic_ok else 'no data'}"
                )
                last_heartbeat = now

            toggle = bool(pose.get("toggle_data_collection", np.array([False]))[0])
            abort = bool(pose.get("toggle_data_abort", np.array([False]))[0])

            if toggle or abort:
                print(f"[Recorder] Signal received — toggle={toggle}  abort={abort}"
                      f"  left_grip={float(pose.get('left_grip', [0])[0]):.2f}"
                      f"  (pose msg #{pose_msg_count})")

            # Abort: discard current episode
            if abort and recording:
                print(f"[Recorder] Episode ABORTED — discarded {len(buf)} frames")
                recording = False
                buf = None

            # Toggle: start or stop recording
            if toggle:
                if not recording:
                    recording = True
                    buf = _EpisodeBuffer()
                    episode_idx += 1
                    print(f"[Recorder] === Recording started (episode {episode_idx}) ===")
                else:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    episode_dir = os.path.join(args.output_dir, f"{ts}_ep{episode_idx:04d}")
                    save_episode(buf, episode_dir)
                    recording = False
                    buf = None

            # Accumulate data while recording
            if recording and buf is not None:
                buf.append(
                    pico=pose,
                    sonic=sonic_holder.get(),
                    images=image_holder.get() if not args.no_images else None,
                )
                if len(buf) % 100 == 0:
                    print(f"[Recorder]   {len(buf)} frames recorded...")

    except KeyboardInterrupt:
        print("\n[Recorder] Interrupted by user.")
        if recording and buf and len(buf) > 0:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_dir = os.path.join(args.output_dir, f"{ts}_ep{episode_idx:04d}_partial")
            print(f"[Recorder] Saving partial episode ({len(buf)} frames)...")
            save_episode(buf, episode_dir)
    finally:
        stop_event.set()
        pose_sock.close()
        ctx.term()
        print("[Recorder] Shutdown complete.")


if __name__ == "__main__":
    main()
