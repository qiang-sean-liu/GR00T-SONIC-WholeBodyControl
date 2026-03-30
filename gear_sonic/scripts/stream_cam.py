"""Stream MuJoCo head_camera to PICO browser as MJPEG over plain HTTP.

Reads frames from the named shared memory block written by run_sim_loop.py
--head_cam and serves them as a standard MJPEG stream. Works in any browser
in flat 2D mode — no WebXR, no TLS cert, no conflict with XRoboToolkit.

Usage
-----
    # Terminal 1 — MuJoCo sim (with head camera enabled):
    python gear_sonic/scripts/run_sim_loop.py --head_cam

    # Terminal 2 — C++ deployment (unchanged):
    bash deploy.sh sim --input-type zmq_manager

    # Terminal 3 — PICO manager (unchanged):
    python gear_sonic/scripts/pico_manager_thread_server.py --manager --vis_vr3pt --vis_smpl

    # Terminal 4 — this script:
    source .venv_teleop/bin/activate
    python gear_sonic/scripts/stream_cam.py

    Open on PICO browser (flat 2D, no VR button needed):
        http://<PC_IP>:8080
"""

import argparse
import io
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from multiprocessing import shared_memory

import numpy as np
from PIL import Image

_HEAD_CAM_SHM_NAME = "pico_head_cam"
_IMG_SHAPE = (480, 640, 3)

# Shared state: the numpy array view into shared memory (set in main())
_img_array: np.ndarray | None = None


def _encode_jpeg(frame: np.ndarray, quality: int = 75) -> bytes:
    img = Image.fromarray(frame, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class _MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request noise

    def do_GET(self):
        if self.path == "/":
            # Serve a minimal HTML page that embeds the stream
            html = (
                b"<html><body style='margin:0;padding:0;background:#000;"
                b"display:flex;justify-content:center;align-items:center;height:100vh'>"
                b"<img src='/stream' style='max-width:100vw;max-height:100vh;object-fit:contain'>"
                b"</body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            interval = 1.0 / 30.0
            try:
                while True:
                    t0 = time.monotonic()
                    frame = _img_array.copy() if _img_array is not None else None
                    if frame is not None and np.any(frame):
                        jpg = _encode_jpeg(frame)
                        header = (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                        )
                        self.wfile.write(header + jpg + b"\r\n")
                        self.wfile.flush()
                    elapsed = time.monotonic() - t0
                    time.sleep(max(0.0, interval - elapsed))
            except (BrokenPipeError, ConnectionResetError):
                pass  # client disconnected
            return

        self.send_response(404)
        self.end_headers()


def _wait_for_shm(name: str, timeout: float) -> shared_memory.SharedMemory:
    print(f"[stream_cam] Waiting for sim to create '{name}' shared memory …")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            shm = shared_memory.SharedMemory(name=name)
            # Don't let Python's resource tracker unlink a shm we don't own.
            try:
                from multiprocessing import resource_tracker as _rt
                _rt.unregister(f"/{name}", "shared_memory")
            except Exception:
                pass
            print(f"[stream_cam] Attached to '{name}' ({_IMG_SHAPE[0]}x{_IMG_SHAPE[1]})")
            return shm
        except FileNotFoundError:
            time.sleep(0.5)
    raise RuntimeError(
        f"Timed out waiting for '{name}' shared memory.\n"
        "Make sure run_sim_loop.py is running with --head_cam."
    )


def main():
    global _img_array

    parser = argparse.ArgumentParser(description="Stream robot head camera to PICO browser (MJPEG)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    parser.add_argument("--wait", type=float, default=30.0,
                        help="Seconds to wait for sim shared memory (default: 30)")
    args = parser.parse_args()

    shm = _wait_for_shm(_HEAD_CAM_SHM_NAME, args.wait)
    _img_array = np.ndarray(_IMG_SHAPE, dtype=np.uint8, buffer=shm.buf)

    server = HTTPServer(("0.0.0.0", args.port), _MJPEGHandler)
    print(
        f"[stream_cam] MJPEG server running\n"
        f"  Open on PICO browser: http://10.40.8.54:{args.port}\n"
        f"  (No 'Enter VR' button — plain 2D stream, XRoboToolkit unaffected)"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        shm.close()
        print("[stream_cam] stopped.")


if __name__ == "__main__":
    main()
