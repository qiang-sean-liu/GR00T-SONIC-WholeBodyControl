# SONIC ZMQ And PICO Image Streaming Implementation

This document describes the implementation for:

1. Receiving PICO / teleop data on the PC via ZMQ.
2. Publishing MuJoCo camera images to the recorder via ZMQ.
3. Streaming MuJoCo head-camera images back to the PICO headset.

There are three related data paths in the collection stack:

1. **PICO / teleop data into the PC over ZMQ**: `pico_manager_thread_server.py` publishes `pose`, `planner`, and `manager_state` messages on port `5556`.
2. **MuJoCo camera images into the recorder over ZMQ**: `run_sim_loop.py` publishes JPEG camera frames on port `5555` (or the configured `--camera-port`), and the recorder subscribes to them.
3. **MuJoCo head-camera image back to PICO**: `stream_cam_xr.py` reads a shared-memory head-camera image and streams H.264 frames to PICO's XRoboToolkit Remote Vision panel. This return path is not the same ZMQ image stream used by the recorder; it uses the Remote Vision command/stream TCP protocol.

## Receiving PICO Data Via ZMQ

`pico_manager_thread_server.py` reads headset/body/controller data from XRoboToolkit, converts it into SONIC-friendly arrays, and publishes structured ZMQ messages. The wire format is:

```text
[topic bytes][2048-byte JSON header][binary payload]
```

The message builder lives in `gear_sonic/utils/teleop/zmq/zmq_planner_sender.py`:

```python
HEADER_SIZE = 2048

def pack_pose_message(pose_data: dict, topic: str = "pose", version: int = 3) -> bytes:
    fields = []
    binary_data = []
    for key, value in pose_data.items():
        if isinstance(value, np.ndarray):
            fields.append({"name": key, "dtype": dtype_str, "shape": list(value.shape)})
            binary_data.append(value.tobytes())

    header_bytes = _build_header(fields, version=version, count=1)
    return topic.encode("utf-8") + header_bytes + b"".join(binary_data)
```

The PICO manager publishes three topics:

| Topic | Contents | Consumer |
|---|---|---|
| `pose` | SMPL joints/pose, body root quaternion, VR 3-point pose, wrist/hand/controller state | C++ deploy and recorder |
| `planner` | planner mode, movement/facing/speed/height, VR 3-point target, hand joints | C++ deploy and recorder |
| `manager_state` | mode/toggle/control state from headset buttons | C++ deploy and recorder |

The recorder subscribes to those messages using a ZMQ `SUB` socket in `record_sonic_teleop.py`:

```python
def _sonic_subscriber(port: int, topic: str, holder: _LatestValue, stop: threading.Event, host: str):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, 10)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.RCVTIMEO, 200)
    sock.connect(f"tcp://{host}:{port}")

    while not stop.is_set():
        try:
            raw = sock.recv()
            parsed = unpack_sonic_message(raw, topic)
            if parsed is not None:
                holder.put(parsed)
        except zmq.Again:
            pass
```

For the normal setup, the recorder connects to `tcp://localhost:5556` and subscribes to `pose`. C++ deploy also subscribes to the same PICO manager stream for live control.

## Publishing MuJoCo Images To The Recorder Via ZMQ

When `run_sim_loop.py` is launched with image publishing enabled, MuJoCo renders camera frames offscreen and the simulator writes them into shared memory. A separate image-publish process reads those frames, JPEG-encodes them, and sends them over ZMQ:

```python
sensor_server = SensorServer()
sensor_server.start_server(port=zmq_port)

image_copies = {
    name: cv2.cvtColor(arr.copy(), cv2.COLOR_RGB2BGR)
    for name, arr in shared_arrays.items()
}
image_msg = ImageMessageSchema(
    timestamps={name: current_time for name in image_copies.keys()},
    images=image_copies,
)
serialized_data = image_msg.serialize()
for camera_name, image_copy in image_copies.items():
    serialized_data[f"{camera_name}"] = ImageUtils.encode_image(image_copy)
sensor_server.send_message(serialized_data)
```

The recorder receives this stream with a separate image subscriber:

```python
def _image_subscriber(port: int, holder: _LatestValue, stop: threading.Event, host: str):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, 5)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVTIMEO, 200)
    sock.connect(f"tcp://{host}:{port}")

    while not stop.is_set():
        try:
            raw = sock.recv()
            result = unpack_image_message(raw)
            if result is not None:
                holder.put(result)  # (images_dict, timestamps_dict)
        except zmq.Again:
            pass
```

This is the camera stream saved into the dataset.

## Streaming MuJoCo Head Camera Back To PICO

The live PICO visual feedback path is implemented by `stream_cam_xr.py`. The simulator creates a named shared-memory block called `pico_head_cam` when launched with `--head_cam`; stereo mode stores a side-by-side image:

```python
self.head_cam_shm = shared_memory.SharedMemory(
    name=head_cam_shm_name,
    create=True,
    size=size,
)
self.head_cam_shm_array = np.ndarray((h, shm_w, 3), dtype=np.uint8, buffer=self.head_cam_shm.buf)
```

`stream_cam_xr.py` attaches to that shared memory, waits for PICO to request a camera stream, then sends H.264 frames back to PICO:

```python
def _recv_framed(sock: socket.socket) -> bytes:
    length = struct.unpack(">I", _recv_exact(sock, 4))[0]
    return _recv_exact(sock, length)

def _send_framed(sock: socket.socket, data: bytes) -> None:
    sock.sendall(struct.pack(">I", len(data)) + data)
```

The command channel listens on port `13579`. PICO sends `OPEN_CAMERA`; the PC parses the requested stream configuration:

```python
cfg = _parse_camera_request(data)
```

Then the PC connects back to PICO's requested stream port (usually `12345`) and pushes H.264 NAL packets:

```python
sock = socket.create_connection((pico_ip, stream_port), timeout=5)
encoder = H264Encoder(width, height, fps, bitrate_kbps)

while not stop_event.is_set():
    frame = img_array.copy()
    if np.any(frame):
        for nal in encoder.encode(frame):
            _send_framed(sock, nal)
```

So the return-image path is:

```text
MuJoCo offscreen head camera
  -> shared memory `pico_head_cam`
  -> stream_cam_xr.py
  -> H.264 Remote Vision TCP stream
  -> PICO headset
```
