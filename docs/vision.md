# pgoggles Vision

pgoggles is an open XR development platform for clip-on smart-glasses modules.
The project centers on ordinary glasses, inexpensive hardware, and SDKs that let
developers stream sensors, run computer-vision pipelines, and render lightweight
overlays without depending on a closed glasses ecosystem.

## Architecture

The platform is organized into three layers:

- `hardware`: reference clip-on module designs, wiring, BOMs, CAD, assembly,
  calibration, and validation notes.
- `sdk`: Python and TypeScript APIs for device connections, camera frames, IMU
  pose samples, overlay rendering, streaming, app events, and simulator support.
- `apps`: examples that demonstrate object detection, captions, navigation
  prompts, QR scanning, remote assist, and opt-in research demos.

The first implementation target is a developer simulator and breadboard module.
Compute runs on a phone, laptop, or edge device first; firmware and onboard
inference can mature after the API contracts are stable.

## Design Principles

- Hardware should be reproducible with commodity parts.
- Simulator mode should let developers build before they own hardware.
- Sensors, displays, and transports should be swappable behind stable interfaces.
- Recording, identification, and biometric workflows must require explicit opt in.
