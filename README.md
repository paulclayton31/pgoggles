# pgoggles

pgoggles is an open XR development platform for clip-on smart-glasses modules.
It combines reference hardware, simulator-first SDKs, and sample apps for camera
frames, IMU pose data, lightweight visual overlays, and computer-vision hooks.

The original face-vector work now lives in `research/identity` as an optional
research package. It is not part of the default install path.

## Workspace

- `docs/`: product vision, hardware MVP, SDK contracts, and privacy policy.
- `packages/python-sdk/`: Python SDK with threaded video pipeline, simulator device contracts, and transport layer.
- `packages/js-sdk/`: early TypeScript SDK and device interfaces.
- `firmware/` and `hardware/`: plans for firmware and the clip-on module.
- `packages/python-sdk/examples/`: runnable Python demos.
- `research/identity/`: optional identity research package.

## Setup

Use Python 3.10 or newer. Run setup from the repository root.

### macOS and Linux

```bash
./setup.sh
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ./packages/python-sdk
```

### Optional identity research

The default setup installs only the base SDK dependencies. To install the legacy
identity research dependencies, opt in explicitly:

```bash
PGOGGLES_INSTALL_IDENTITY_RESEARCH=1 ./setup.sh
```

Identity research code also requires runtime opt-in:

```bash
PGOGGLES_IDENTITY_RESEARCH_ENABLED=1 python -m research.identity.db.server
```

## Video Pipeline

The Python SDK ships a threaded video pipeline (`pgoggles.pipeline`) that chains
camera capture, YOLO inference, and overlay rendering across independent threads:

- **`CameraSource`** — captures frames from a webcam (or, in production, an
  ESP32-S3 OV2640 transport) and publishes the freshest `FramePacket` to a
  bounded queue.
- **`AnalysisPipeline`** — runs YOLO object detection on a background thread,
  throttled to a configurable analysis rate, and exposes the latest
  `OverlayState` (bounding boxes, inference latency, scene-change score).
- **`draw_overlay`** — composes detection boxes and a HUD (FPS, analysis rate,
  inference time, analyzing indicator) onto a frame via OpenCV.
- **`DisplayWindow`** — drives the display loop, pulling the latest frame and
  overlay state at a target FPS until the user presses `q`.

A webcam demo is available at `packages/python-sdk/examples/webcam_demo.py`.
After placing `yolov8n.onnx` in `packages/python-sdk`, run the demo with your
virtual environment active:

```bash
cd packages/python-sdk
python examples/webcam_demo.py
```

The pipeline is transport-agnostic: swap `CameraSource` for an ESP32-backed
source behind the same `start/stop/latest_frame/queue` boundary and the rest of
the pipeline remains unchanged.

### YOLO Model Export

The pipeline expects an ONNX model file (default `yolov8n.onnx`) at the SDK
root. Export it once using the `ultralytics` package:

```bash
cd packages/python-sdk
pip install ultralytics
yolo export model=yolov8n.pt format=onnx
```

The `.pt` weights and `.onnx` model are ignored by Git. To keep the runtime
environment lean, you can uninstall `ultralytics` after exporting:

```bash
pip uninstall ultralytics -y
```

Alternatively, install it as an optional dependency:

```bash
pip install ".[export]"
```

## Direction

The first product target is a low-cost module that clips onto ordinary glasses:
a camera, IMU, simple monocular display, and USB-C or wireless transport. Heavy
CV inference runs on a phone, laptop, or edge device first. The simulator should
let developers build apps before they own hardware.

See [docs/vision.md](docs/vision.md) and [docs/sdk-api.md](docs/sdk-api.md) for
the architecture and API contracts.
