"""Remote Vision Server for PICO XRoboToolkit.

Listens on port 13579 for OPEN_CAMERA from PICO, then connects back to
PICO:12345 and pushes JPEG frames from the sim head-camera shared memory.

Usage
-----
    # Terminal 1 — sim with head camera:
    python gear_sonic/scripts/run_sim_loop.py --head_cam

    # Terminal 2 — this server:
    source .venv_teleop/bin/activate
    python gear_sonic/scripts/remote_vision_server.py

Then open XRoboToolkit on PICO → Remote Vision → enter PC IP → Connect.
"""

import io
import socket
import struct
import threading
import time

import numpy as np
from PIL import Image

# ── shared-memory config (same as stream_cam.py) ──────────────────────────────
_SHM_NAME = "pico_head_cam"
_IMG_SHAPE = (480, 640, 3)

CMD_PORT = 13579
STREAM_PORT = 12345
TARGET_FPS = 30
JPEG_QUALITY = 80


def _encode_jpeg(frame: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame, mode="RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _open_shm():
    from multiprocessing import shared_memory, resource_tracker as _rt
    print(f"[vision] waiting for shared memory '{_SHM_NAME}' …")
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            shm = shared_memory.SharedMemory(name=_SHM_NAME)
            try:
                _rt.unregister(f"/{_SHM_NAME}", "shared_memory")
            except Exception:
                pass
            arr = np.ndarray(_IMG_SHAPE, dtype=np.uint8, buffer=shm.buf)
            print(f"[vision] attached to '{_SHM_NAME}'")
            return shm, arr
        except FileNotFoundError:
            time.sleep(0.5)
    raise RuntimeError(
        f"Timed out waiting for '{_SHM_NAME}'.\n"
        "Run: python gear_sonic/scripts/run_sim_loop.py --head_cam"
    )


def _parse_open_camera(data: bytes) -> str | None:
    """Extract PICO IP from OPEN_CAMERA message.

    Message layout (69 bytes):
      [4B LE total_len] [4B LE cmd_len] [cmd_len bytes cmd] ['.' 0x2e]
      [3B pad] [2B magic 0xCAFE] [...params...] [1B ip_len] [ip_bytes]
    """
    try:
        # IP length byte is immediately before the IP string at the end
        # "VR" (0x5652) precedes ip_len
        vr_pos = data.rfind(b"VR")
        if vr_pos == -1:
            return None
        ip_len = data[vr_pos + 2]
        ip_bytes = data[vr_pos + 3: vr_pos + 3 + ip_len]
        return ip_bytes.decode("ascii")
    except Exception:
        return None


def _make_frame(jpg: bytes, fmt: int) -> bytes:
    """Try different frame formats to find what PICO accepts."""
    if fmt == 0:
        # Raw JPEG, no header
        return jpg
    elif fmt == 1:
        # 4-byte BE length + JPEG
        return struct.pack(">I", len(jpg)) + jpg
    elif fmt == 2:
        # 4-byte LE length + JPEG
        return struct.pack("<I", len(jpg)) + jpg
    elif fmt == 3:
        # 0xCAFE (BE) + 4-byte BE length + JPEG
        return b"\xca\xfe" + struct.pack(">I", len(jpg)) + jpg
    elif fmt == 4:
        # 0xCAFE (BE) + 4-byte LE length + JPEG
        return b"\xca\xfe" + struct.pack("<I", len(jpg)) + jpg
    elif fmt == 5:
        # 2-byte LE width + 2-byte LE height + 4-byte LE length + JPEG
        h, w = _IMG_SHAPE[:2]
        return struct.pack("<HHI", w, h, len(jpg)) + jpg
    elif fmt == 6:
        # 0xCAFE + width(2B LE) + height(2B LE) + length(4B LE) + JPEG
        h, w = _IMG_SHAPE[:2]
        return b"\xca\xfe" + struct.pack("<HHI", w, h, len(jpg)) + jpg
    return jpg

_FMT_NAMES = {
    0: "raw JPEG",
    1: "4B-BE-len + JPEG",
    2: "4B-LE-len + JPEG",
    3: "CAFE + 4B-BE-len + JPEG",
    4: "CAFE + 4B-LE-len + JPEG",
    5: "W+H + 4B-LE-len + JPEG",
    6: "CAFE + W+H + 4B-LE-len + JPEG",
}

# Shared queue: cmd handler puts pico_ip here; stream handler picks it up
_stream_queue: list = []
_stream_lock = threading.Lock()


def _stream_conn(sock: socket.socket, fmt: int, img_arr: np.ndarray):
    """Send JPEG frames on an already-connected socket."""
    interval = 1.0 / TARGET_FPS
    sent = 0
    try:
        while True:
            t0 = time.monotonic()
            frame = img_arr.copy()
            if np.any(frame):
                jpg = _encode_jpeg(frame)
                sock.sendall(_make_frame(jpg, fmt))
                sent += 1
                if sent == 1:
                    print(f"[vision] fmt {fmt}: first frame sent …", flush=True)
                if sent % 150 == 0:
                    print(f"[vision] fmt {fmt}: {sent} frames — PICO still connected!", flush=True)
            elapsed = time.monotonic() - t0
            # No sleep for first 10 frames — keep stream tight to prevent PICO timeout
            if sent >= 10:
                time.sleep(max(0.0, interval - elapsed))
    except (BrokenPipeError, ConnectionResetError, OSError):
        print(f"[vision] fmt {fmt} ({_FMT_NAMES[fmt]}): disconnected after {sent} frames", flush=True)
    finally:
        sock.close()
    return sent


def _stream_server(img_arr: np.ndarray):
    """Listen on STREAM_PORT; for each PICO connection try all frame formats."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", STREAM_PORT))
    srv.listen(5)
    print(f"[vision] stream server listening on :{STREAM_PORT}", flush=True)
    while True:
        conn, addr = srv.accept()
        print(f"[vision] PICO stream connected from {addr}", flush=True)
        for fmt in range(len(_FMT_NAMES)):
            print(f"[vision] trying fmt {fmt}: {_FMT_NAMES[fmt]}", flush=True)
            sent = _stream_conn(conn, fmt, img_arr)
            if sent > 5:
                print(f"[vision] fmt {fmt} works! ({sent} frames)", flush=True)
                break
            if sent == 0:
                break  # PICO closed before we sent anything — stop trying
            # Need a new connection for next format attempt
            print(f"[vision] reconnecting to try next format …", flush=True)
            try:
                # Try connecting TO PICO for next attempt
                with _stream_lock:
                    pico_ip = _stream_queue[-1] if _stream_queue else addr[0]
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.settimeout(3)
                conn.connect((pico_ip, STREAM_PORT))
            except Exception as e:
                print(f"[vision] cannot reconnect: {e}", flush=True)
                break


def _handle_cmd(conn, addr, img_arr):
    conn.settimeout(3)
    chunks = []
    try:
        while True:
            d = conn.recv(4096)
            if not d:
                break
            chunks.append(d)
    except socket.timeout:
        pass

    data = b"".join(chunks)
    if b"OPEN_CAMERA" not in data:
        print(f"[vision] unexpected command from {addr}: {data[:32].hex()}")
        conn.close()
        return

    pico_ip = _parse_open_camera(data)
    if pico_ip is None:
        pico_ip = addr[0]
        print(f"[vision] could not parse PICO IP, using {pico_ip}")

    with _stream_lock:
        _stream_queue.append(pico_ip)

    print(f"[vision] OPEN_CAMERA from {pico_ip} — sending ACK, then connecting to PICO:{STREAM_PORT}", flush=True)

    # Send ACK before closing command connection
    try:
        conn.send(b"\x00\x00\x00\x02OK")
    except Exception:
        pass
    conn.close()

    # Also try connecting TO PICO:12345 in a thread (fallback if PICO doesn't connect to us)
    def _try_push():
        time.sleep(0.3)  # give PICO time to open its port after ACK

        # Phase 1: test if PICO accepts repeated per-connection single frames (fmt 0)
        print("[vision] testing per-connection mode (fmt 0, up to 30 reconnects) …", flush=True)
        success_count = 0
        for attempt in range(30):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1)
                s.connect((pico_ip, STREAM_PORT))
            except Exception as e:
                if attempt == 0:
                    print(f"[vision] PICO:{STREAM_PORT} refused on attempt {attempt}: {e}", flush=True)
                    break
                # PICO not ready yet — wait and retry
                time.sleep(0.1)
                continue
            frame = img_arr.copy()
            if np.any(frame):
                jpg = _encode_jpeg(frame)
                try:
                    s.sendall(_make_frame(jpg, 0))
                    success_count += 1
                except Exception:
                    pass
            s.close()
            if attempt % 5 == 0:
                print(f"[vision] per-connection: {success_count}/{attempt+1} frames sent", flush=True)
            time.sleep(1.0 / TARGET_FPS)

        if success_count > 5:
            print(f"[vision] per-connection mode works ({success_count} frames)! Switching to continuous reconnect loop.", flush=True)
            # Switch to continuous mode
            while True:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(1)
                    s.connect((pico_ip, STREAM_PORT))
                    frame = img_arr.copy()
                    if np.any(frame):
                        s.sendall(_make_frame(img_arr.copy(), 0))
                    s.close()
                except Exception:
                    pass
                time.sleep(1.0 / TARGET_FPS)

        # Phase 2: try multi-frame formats (need PICO to reconnect via new OPEN_CAMERA)
        print("[vision] per-connection mode failed. Try reconnecting from PICO for multi-frame format test.", flush=True)

    threading.Thread(target=_try_push, daemon=True).start()


def main():
    shm, img_arr = _open_shm()

    # Stream server: listens in case PICO connects to US on :12345
    threading.Thread(target=_stream_server, args=(img_arr,), daemon=True).start()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", CMD_PORT))
    srv.listen(5)
    print(f"[vision] cmd server listening on :{CMD_PORT}", flush=True)
    print(f"[vision] On PICO: XRoboToolkit → Remote Vision → enter PC IP → Connect", flush=True)

    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(
                target=_handle_cmd, args=(conn, addr, img_arr), daemon=True
            ).start()
    except KeyboardInterrupt:
        print("[vision] stopped")
    finally:
        shm.close()


if __name__ == "__main__":
    main()
