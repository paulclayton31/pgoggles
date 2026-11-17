# Hardware MVP

The first pgoggles hardware target is a clip-on module for ordinary glasses. The
MVP keeps optics and compute intentionally simple so the SDK can mature around a
real device without requiring custom silicon or expensive manufacturing.

## Reference Module

- Monocular microdisplay or simple transparent display.
- USB camera, phone camera stream, or small embedded camera module.
- IMU for head motion and orientation sampling.
- USB-C first transport, with BLE or Wi-Fi as follow-on options.
- Battery and charging board for untethered smoke testing.
- 3D-printable clip mount with adjustable display and camera positioning.

## Prototype Phases

1. Desktop simulator with sample video and overlay rendering.
2. Breadboard camera plus IMU streaming to the Python SDK.
3. Clip-on enclosure with stable cable routing and repeatable mounting.
4. Firmware transport protocol and hardware acceptance checklist.

## Validation Targets

- Camera stream latency.
- IMU sample rate and timestamp consistency.
- Display update rate.
- Battery runtime.
- Thermal behavior during sustained streaming.
