# SDK API

The SDK exposes a small device contract that works with both real hardware and
the simulator.

## Core Device API

- `connect()`: open a simulator, USB, BLE, Wi-Fi, or custom transport.
- `get_frame()`: return the latest camera frame with timestamp and frame ID.
- `get_pose()`: return the latest IMU pose or orientation sample.
- `render_overlay()`: submit text, icons, boxes, reticles, notifications, or
  images for display.
- `send_event()`: send app or input events to the device.
- `start_stream()`: begin continuous frames and sensor samples.
- `stop_stream()`: stop continuous streaming and release resources.

The initial implementation is a skeleton. `SimulatorDevice` is functional for
application development, while USB, BLE, and Wi-Fi transports are explicit
placeholders until the hardware protocol is defined.

## Message Schema

Frame messages should include `frame_id`, `timestamp_ms`, `width`, `height`,
`format`, and payload metadata. Sensor samples should include `timestamp_ms`,
coordinate-system metadata, and raw or fused pose values. Overlay messages should
include `overlay_id`, `frame_id` when anchored to a frame, `timestamp_ms`, type,
geometry, style, and expiration.

Python dataclasses live in `pgoggles.messages`. The TypeScript SDK mirrors the
same shape with message types in `packages/js-sdk/src/index.ts`.

## Display Primitives

The Python SDK includes convenience constructors in `pgoggles.overlays` for:

- `text`
- `bounding_box`
- `reticle`
- `notification`

Additional primitives, including icons, images, and layout regions, are reserved
in the shared overlay type enum and can be filled in once display constraints are
known.

## Plugin Lifecycle

Apps consume frame and pose messages and emit overlays. Plugins should support:

- `setup(context)`
- `on_frame(frame, context)`
- `on_pose(pose, context)`
- `on_event(event, context)`
- `teardown(context)`

Plugins must handle disconnected devices and dropped frames without crashing the
runtime.

The Python SDK provides `BasePlugin`, `PluginContext`, and `PluginRuntime` as the
minimal async lifecycle skeleton.
