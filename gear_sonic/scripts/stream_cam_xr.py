"""Stream MuJoCo head_camera to Pico XRoboToolkit Remote Vision panel.

Implements the OrinVideoSender wire protocol so the Pico's built-in
"Remote Vision" panel receives the MuJoCo head camera — no browser needed,
no focus conflict with XRoboToolkit.

Protocol (reverse-engineered from XRoboToolkit-Orin-Video-Sender):
  Command port (default 13579): PC listens; Pico connects and sends
    OPEN_CAMERA / CLOSE_CAMERA commands.
  Streaming port (default 12345): Pico listens; on OPEN_CAMERA the PC
    connects back and pushes H.264 frames as [4-byte BE length][NAL data].

Usage
-----
    # Terminal 1 — MuJoCo sim with head camera:
    python gear_sonic/scripts/run_sim_loop.py --head_cam

    # Terminal 2 — this script:
    source .venv_teleop/bin/activate
    python gear_sonic/scripts/stream_cam_xr.py

    On Pico XRoboToolkit → Remote Vision panel:
        Camera source IP : <this PC's IP>
        Command port     : 13579
        Streaming port   : 12345

Requires: pip install av (PyAV, for software H.264 encoding)
"""

import argparse
import socket
import struct
import threading
import time
from multiprocessing import shared_memory

import av
import fractions
import numpy as np

_MAX_BITRATE_KBPS = 20_000  # cap at 20 Mbps regardless of what Pico requests

_HEAD_CAM_SHM_NAME = "pico_head_cam"
_IMG_SHAPE = (480, 1280, 3)  # height, width, channels — stereo SBS: left(640) | right(640)


# ---------------------------------------------------------------------------
# Protocol helpers
# ---------------------------------------------------------------------------

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed by peer")
        buf.extend(chunk)
    return bytes(buf)


def _recv_framed(sock: socket.socket) -> bytes:
    """Read one message: [4-byte BE length][payload]."""
    length = struct.unpack(">I", _recv_exact(sock, 4))[0]
    return _recv_exact(sock, length)


def _send_framed(sock: socket.socket, data: bytes) -> None:
    """Write one message: [4-byte BE length][payload]."""
    sock.sendall(struct.pack(">I", len(data)) + data)


def _parse_compact_str(data: bytes, offset: int) -> tuple[str, int]:
    """Read a 1-byte-length-prefixed string. Returns (value, next_offset)."""
    length = data[offset]
    value = data[offset + 1: offset + 1 + length].decode("utf-8", errors="replace")
    return value, offset + 1 + length


def _parse_network_data_protocol(msg: bytes) -> tuple[str, bytes]:
    """Parse NetworkDataProtocol → (command_str, data_bytes).

    Layout:
        [4-byte LE cmd_length][cmd_string]
        [4-byte LE data_length][data]
    """
    cmd_len = struct.unpack_from("<I", msg, 0)[0]
    cmd = msg[4: 4 + cmd_len].rstrip(b"\x00").decode()
    data_len = struct.unpack_from("<I", msg, 4 + cmd_len)[0]
    data = msg[4 + cmd_len + 4: 4 + cmd_len + 4 + data_len]
    return cmd, data


def _parse_camera_request(payload: bytes) -> dict:
    """Parse CameraRequestData from an OPEN_CAMERA payload.

    Layout:
        0       2   magic 0xCA 0xFE
        2       1   version (expect 1)
        3      28   7 x int32_le: width, height, fps, bitrate_kbps,
                    enable_hevc, render_mode, stream_port
       31       *   compact string: camera_type
       31+*     *   compact string: ip_address  (Pico's IP to stream back to)
    """
    if len(payload) < 33:
        raise ValueError(f"OPEN_CAMERA payload too short ({len(payload)} bytes)")
    if payload[0:2] != b"\xca\xfe":
        raise ValueError(f"Bad magic bytes: {payload[0:2].hex()}")
    version = payload[2]
    (width, height, fps, bitrate_kbps,
     enable_hevc, render_mode, stream_port) = struct.unpack_from("<7i", payload, 3)
    camera_type, off = _parse_compact_str(payload, 31)
    ip_address, _ = _parse_compact_str(payload, off)
    return {
        "version": version,
        "width": width,
        "height": height,
        "fps": fps,
        "bitrate_kbps": bitrate_kbps,
        "enable_hevc": bool(enable_hevc),
        "render_mode": render_mode,
        "stream_port": stream_port,
        "camera_type": camera_type,
        "ip_address": ip_address,
    }


# ---------------------------------------------------------------------------
# H.264 encoder (PyAV / libx264)
# ---------------------------------------------------------------------------

class H264Encoder:
    def __init__(self, width: int, height: int, fps: int, bitrate_kbps: int):
        self.width = width
        self.height = height
        self.fps = fps
        self._pts = 0

        self._ctx = av.CodecContext.create("libx264", "w")
        self._ctx.width = width
        self._ctx.height = height
        self._ctx.time_base = fractions.Fraction(1, fps)
        self._ctx.framerate = fractions.Fraction(fps, 1)
        self._ctx.pix_fmt = "yuv420p"
        capped_kbps = min(bitrate_kbps, _MAX_BITRATE_KBPS)
        self._ctx.bit_rate = capped_kbps * 1000
        self._ctx.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "profile": "baseline",
        }
        self._ctx.open()

    def encode(self, frame_rgb: np.ndarray) -> list[bytes]:
        """Encode one RGB (H×W×3) frame. Returns list of H.264 packet bytes."""
        if frame_rgb.shape[1] != self.width or frame_rgb.shape[0] != self.height:
            import cv2
            frame_rgb = cv2.resize(frame_rgb, (self.width, self.height))
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        av_frame = av_frame.reformat(format="yuv420p")
        av_frame.pts = self._pts
        self._pts += 1
        return [bytes(pkt) for pkt in self._ctx.encode(av_frame)]

    def flush(self) -> list[bytes]:
        return [bytes(pkt) for pkt in self._ctx.encode(None)]


# ---------------------------------------------------------------------------
# Streaming worker
# ---------------------------------------------------------------------------

def _stream_worker(
    pico_ip: str,
    stream_port: int,
    width: int,
    height: int,
    fps: int,
    bitrate_kbps: int,
    img_array: np.ndarray,
    stop_event: threading.Event,
) -> None:
    """Connect to Pico on stream_port and push H.264 frames until stop_event."""
    print(f"[stream_cam_xr] Connecting to {pico_ip}:{stream_port} …")
    try:
        sock = socket.create_connection((pico_ip, stream_port), timeout=5)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError as exc:
        print(f"[stream_cam_xr] Cannot connect to {pico_ip}:{stream_port}: {exc}")
        return

    print(
        f"[stream_cam_xr] Streaming {width}x{height}@{fps}fps "
        f"({bitrate_kbps} kbps) to {pico_ip}:{stream_port}"
    )
    encoder = H264Encoder(width, height, fps, bitrate_kbps)
    interval = 1.0 / fps

    try:
        while not stop_event.is_set():
            t0 = time.monotonic()
            frame = img_array.copy()
            if np.any(frame):
                for nal in encoder.encode(frame):
                    _send_framed(sock, nal)
            elapsed = time.monotonic() - t0
            sleep_for = interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    except (BrokenPipeError, ConnectionResetError, OSError):
        print("[stream_cam_xr] Streaming connection lost.")
    finally:
        for nal in encoder.flush():
            try:
                _send_framed(sock, nal)
            except OSError:
                break
        sock.close()
        print("[stream_cam_xr] Stream worker stopped.")


# ---------------------------------------------------------------------------
# Command server (listens for OPEN_CAMERA / CLOSE_CAMERA from Pico)
# ---------------------------------------------------------------------------

def _handle_connection(
    conn: socket.socket,
    addr: tuple,
    img_array: np.ndarray,
    stream_thread_ref: list,   # mutable 1-element list used as a box
    stop_event: threading.Event,
) -> None:
    print(f"[stream_cam_xr] Pico connected from {addr}")
    try:
        while True:
            msg = _recv_framed(conn)
            cmd, data = _parse_network_data_protocol(msg)
            print(f"[stream_cam_xr] Command received: {cmd}")

            if cmd == "OPEN_CAMERA":
                try:
                    cfg = _parse_camera_request(data)
                except ValueError as exc:
                    print(f"[stream_cam_xr] Bad OPEN_CAMERA payload: {exc}")
                    continue
                print(f"[stream_cam_xr] OPEN_CAMERA cfg: {cfg}")

                # Stop any existing stream first
                stop_event.set()
                old = stream_thread_ref[0]
                if old and old.is_alive():
                    old.join(timeout=3)
                stop_event.clear()

                t = threading.Thread(
                    target=_stream_worker,
                    args=(
                        cfg["ip_address"],
                        cfg["stream_port"],
                        cfg["width"],
                        cfg["height"],
                        cfg["fps"],
                        cfg["bitrate_kbps"],
                        img_array,
                        stop_event,
                    ),
                    daemon=True,
                )
                stream_thread_ref[0] = t
                t.start()

            elif cmd == "CLOSE_CAMERA":
                stop_event.set()
                old = stream_thread_ref[0]
                if old and old.is_alive():
                    old.join(timeout=3)
                stop_event.clear()
                stream_thread_ref[0] = None
                print("[stream_cam_xr] Camera closed by Pico.")

            else:
                print(f"[stream_cam_xr] Unknown command: {cmd!r}")

    except (ConnectionError, struct.error) as exc:
        print(f"[stream_cam_xr] Connection dropped: {exc}")
    finally:
        conn.close()


def run_command_server(cmd_port: int, img_array: np.ndarray) -> None:
    """Accept connections from the Pico on cmd_port indefinitely."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", cmd_port))
    srv.listen(1)
    print(
        f"[stream_cam_xr] Command server ready on port {cmd_port}.\n"
        f"  On Pico XRoboToolkit → Remote Vision:\n"
        f"    Camera source IP : <this machine's LAN IP>\n"
        f"    Command port     : {cmd_port}\n"
        f"    Streaming port   : 12345  (Pico default)\n"
    )

    stream_thread_ref: list = [None]
    stop_event = threading.Event()

    try:
        while True:
            conn, addr = srv.accept()
            # Stop any previous stream when a new Pico connects
            stop_event.set()
            old = stream_thread_ref[0]
            if old and old.is_alive():
                old.join(timeout=3)
            stop_event.clear()

            _handle_connection(conn, addr, img_array, stream_thread_ref, stop_event)

    except KeyboardInterrupt:
        print("\n[stream_cam_xr] Interrupted.")
    finally:
        stop_event.set()
        old = stream_thread_ref[0]
        if old and old.is_alive():
            old.join(timeout=3)
        srv.close()
        print("[stream_cam_xr] Server stopped.")


# ---------------------------------------------------------------------------
# Shared memory setup + entry point
# ---------------------------------------------------------------------------

def _wait_for_shm(name: str, timeout: float) -> shared_memory.SharedMemory:
    expected = _IMG_SHAPE[0] * _IMG_SHAPE[1] * _IMG_SHAPE[2]
    print(f"[stream_cam_xr] Waiting for '{name}' ({_IMG_SHAPE[1]}x{_IMG_SHAPE[0]}, {expected} bytes) …")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            shm = shared_memory.SharedMemory(name=name)
            if shm.size != expected:
                shm.close()
                print(
                    f"[stream_cam_xr] '{name}' exists but size {shm.size} != {expected} "
                    f"(stale mono shm?). Waiting for sim restart …"
                )
                time.sleep(0.5)
                continue
            # Prevent Python's resource tracker from unlinking memory we don't own.
            try:
                from multiprocessing import resource_tracker as _rt
                _rt.unregister(f"/{name}", "shared_memory")
            except Exception:
                pass
            print(f"[stream_cam_xr] Attached to '{name}' ({_IMG_SHAPE[1]}x{_IMG_SHAPE[0]})")
            return shm
        except FileNotFoundError:
            time.sleep(0.5)
    raise RuntimeError(
        f"Timed out waiting for '{name}' shared memory.\n"
        "Make sure run_sim_loop.py is running with --head_cam."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream MuJoCo head camera to Pico XRoboToolkit Remote Vision"
    )
    parser.add_argument(
        "--cmd-port", type=int, default=13579,
        help="Command port the Pico connects to (default: 13579)"
    )
    parser.add_argument(
        "--wait", type=float, default=30.0,
        help="Seconds to wait for sim shared memory before giving up (default: 30)"
    )
    args = parser.parse_args()

    shm = _wait_for_shm(_HEAD_CAM_SHM_NAME, args.wait)
    img_array = np.ndarray(_IMG_SHAPE, dtype=np.uint8, buffer=shm.buf)

    try:
        run_command_server(args.cmd_port, img_array)
    finally:
        shm.close()
        print("[stream_cam_xr] Done.")


if __name__ == "__main__":
    main()
