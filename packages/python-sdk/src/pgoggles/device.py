"""Core simulator-friendly device contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

from .messages import DeviceEvent, DeviceStatus, Frame, Pose, now_ms


class Device(Protocol):
    connected: bool
    streaming: bool

    def connect(self) -> "Device": ...

    def get_frame(self) -> Frame: ...

    def get_pose(self) -> Pose: ...

    def render_overlay(self, overlay: "Overlay") -> None: ...

    def send_event(self, event: DeviceEvent) -> None: ...

    def start_stream(self) -> None: ...

    def stop_stream(self) -> None: ...

    def status(self) -> DeviceStatus: ...


@dataclass(frozen=True)
class StreamSample:
    frame: Frame | None = None
    pose: Pose | None = None
    event: DeviceEvent | None = None


class DisconnectedDeviceError(RuntimeError):
    pass


class TransportUnavailableError(RuntimeError):
    pass


@dataclass
class SimulatorDevice:
    """Minimal in-memory device for SDK and example development."""

    width: int = 1280
    height: int = 720
    frame_format: str = "rgb"
    transport_name: str = "simulator"
    connected: bool = False
    streaming: bool = False
    overlays: list["Overlay"] = field(default_factory=list)
    events: list[DeviceEvent] = field(default_factory=list)
    _frame_counter: int = 0

    def connect(self) -> "SimulatorDevice":
        self.connected = True
        return self

    def get_frame(self) -> Frame:
        self._require_connected()
        self._frame_counter += 1
        return Frame(
            frame_id=f"sim-{self._frame_counter}",
            timestamp_ms=now_ms(),
            width=self.width,
            height=self.height,
            format=self.frame_format,
            metadata={"source": "simulator"},
        )

    def get_pose(self) -> Pose:
        self._require_connected()
        return Pose(
            timestamp_ms=now_ms(),
            orientation=(0.0, 0.0, 0.0, 1.0),
            position=(0.0, 0.0, 0.0),
            metadata={"source": "simulator"},
        )

    def render_overlay(self, overlay: "Overlay") -> None:
        self._require_connected()
        self.overlays.append(overlay)

    def send_event(self, event: DeviceEvent) -> None:
        self._require_connected()
        self.events.append(event)

    def start_stream(self) -> None:
        self._require_connected()
        self.streaming = True

    def stop_stream(self) -> None:
        self.streaming = False

    def status(self) -> DeviceStatus:
        return DeviceStatus(
            connected=self.connected,
            streaming=self.streaming,
            transport=self.transport_name,
        )

    async def stream(self) -> AsyncIterator[StreamSample]:
        self.start_stream()
        while self.streaming:
            yield StreamSample(frame=self.get_frame(), pose=self.get_pose())

    def _require_connected(self) -> None:
        if not self.connected:
            raise DisconnectedDeviceError("device is not connected")


from .messages import Overlay
