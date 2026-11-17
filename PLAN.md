# Open XR Optics SDK + Clip-On Module Plan

## Summary
Build `pgoggles` into an open-source XR development platform centered on a low-cost clip-on module for ordinary glasses, plus a modular SDK that lets developers stream sensor data, run CV pipelines, and render lightweight visual overlays without buying into Meta’s closed ecosystem.

The current repo should pivot from “face-vector backend” into a broader platform. Existing face recognition and vector database work becomes an optional research package, not the core product.

Context references: [Project North Star](https://docs.projectnorthstar.org/project-north-star/getting-started/about), [Project North Star GitHub](https://github.com/leapmotion/ProjectNorthStar), [OpenARK](https://github.com/augcog/OpenARK), [Meta Ray-Ban smart glasses overview](https://about.fb.com/news/2023/09/new-ray-ban-meta-smart-glasses/).

## Key Changes
- Reframe the project around three layers:
  - `hardware`: open clip-on reference design, BOM, firmware, wiring, CAD, assembly docs.
  - `sdk`: Python/TypeScript APIs for camera frames, IMU data, display primitives, streaming, CV hooks, and device control.
  - `apps`: sample apps showing object detection, captions, navigation prompts, QR scanning, remote assist, and optional identity research demos.
- Keep the first hardware MVP intentionally small:
  - Clip-on display module for regular glasses.
  - Camera input, IMU, simple microdisplay or monocular overlay, BLE/Wi-Fi or USB-C transport.
  - Phone/laptop/edge-device compute first; do not require onboard heavy inference in v1.
- Treat optics as swappable:
  - Define a hardware abstraction for display drivers, camera sources, sensors, and input devices.
  - Support a simulator mode so developers can build without hardware.
  - Document compatibility paths for Project North Star-style headsets and commodity camera/IMU boards.

## SDK + Interfaces
- Create a core device API:
  - `connect()`, `get_frame()`, `get_pose()`, `render_overlay()`, `send_event()`, `start_stream()`, `stop_stream()`.
  - Standard message schema for frames, sensor samples, overlays, device status, and app events.
- Create CV integration points:
  - OpenCV frame pipeline.
  - Optional adapters for ONNX Runtime, MediaPipe-style pipelines, FAISS/pgvector retrieval, and object/person detectors.
  - Async plugin system where apps consume frames and emit overlays.
- Create display primitives:
  - Text, icon, bounding box, reticle, notification, image, and simple layout regions.
  - Latency-aware rendering with timestamps and frame IDs.
- Move current face-vector code into an optional package:
  - Name it something like `pgoggles-research-identity`.
  - Require explicit opt-in configuration.
  - Use consented/local datasets by default.
  - Keep scraping out of the default install and default examples.

## Hardware Roadmap
- Phase 1: Developer simulator and breadboard prototype
  - USB camera or phone camera stream.
  - Desktop/mobile simulator for overlays.
  - Basic Python SDK and sample apps.
- Phase 2: Clip-on module dev kit
  - ESP32-S3 or similar microcontroller for transport/control.
  - Small display module, camera, IMU, battery, charging board, 3D-printable mount.
  - Firmware exposing a stable transport protocol.
- Phase 3: Reference hardware release
  - CAD files, PCB files if needed, BOM, assembly guide, firmware flashing guide, calibration guide.
  - Hardware test jig and acceptance checklist.
- Phase 4: Ecosystem
  - App template, package registry conventions, community examples, contribution docs.
  - Compatibility docs for alternate displays, cameras, compute boards, and headset builds.

## Repo Restructure
- Add top-level docs:
  - `docs/vision.md`: product vision and architecture.
  - `docs/hardware-mvp.md`: clip-on module reference design.
  - `docs/sdk-api.md`: SDK contracts and transport schema.
  - `docs/privacy-safety.md`: identity, recording, consent, and data handling policy.
- Restructure implementation over time:
  - `packages/python-sdk`
  - `packages/js-sdk`
  - `firmware`
  - `hardware`
  - `examples`
  - `research/identity`
- Replace duplicate setup scripts with one workspace setup path and add a proper README that explains the new direction.

## Test Plan
- SDK tests:
  - Mock device connection.
  - Frame streaming.
  - Overlay serialization.
  - Plugin lifecycle.
  - Error handling for disconnected devices and dropped frames.
- Simulator tests:
  - Render text, icons, boxes, and notifications over sample video frames.
  - Verify timestamp/frame alignment.
- Hardware validation:
  - Camera stream latency.
  - IMU sample rate.
  - Display update rate.
  - Battery/runtime smoke test.
  - Thermal check during sustained streaming.
- Privacy tests:
  - Default install contains no scraper execution path.
  - Identity research module requires explicit opt-in.
  - Sample apps use local/test data only.

## Assumptions
- First product target is a clip-on module for ordinary glasses.
- The original face recognition work should survive only as a separate optional research module.
- v1 prioritizes developer accessibility over perfect optics.
- Heavy CV inference runs on a phone, laptop, or edge device first; embedded inference can come later.
- The platform should be open-source-first: permissive SDK license, open hardware files, reproducible BOM, and no dependency on proprietary Meta APIs.
