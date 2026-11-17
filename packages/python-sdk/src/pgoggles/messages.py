"""Message schemas shared by devices, transports, and app plugins."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from time import time
from typing import Any


def now_ms() -> int:
    return int(time() * 1000)


class FrameFormat(str, Enum):
    RGB = "rgb"
    BGR = "bgr"
    GRAYSCALE = "grayscale"
    JPEG = "jpeg"
    H264 = "h264"


class OverlayType(str, Enum):
    TEXT = "text"
    ICON = "icon"
    BOUNDING_BOX = "bounding_box"
    RETICLE = "reticle"
    NOTIFICATION = "notification"
    IMAGE = "image"
    REGION = "region"


class EventType(str, Enum):
    INPUT = "input"
    APP = "app"
    STATUS = "status"
    CONTROL = "control"


@dataclass(frozen=True)
class Frame:
    frame_id: str
    timestamp_ms: int
    width: int
    height: int
    format: str = FrameFormat.RGB.value
    data: bytes | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message(self, include_data: bool = False) -> dict[str, Any]:
        message = asdict(self)
        if not include_data:
            message.pop("data", None)
        return message


@dataclass(frozen=True)
class Pose:
    timestamp_ms: int
    orientation: tuple[float, float, float, float]
    position: tuple[float, float, float] | None = None
    coordinate_system: str = "right-handed-y-up"
    angular_velocity: tuple[float, float, float] | None = None
    linear_acceleration: tuple[float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Overlay:
    overlay_id: str
    type: str
    payload: dict[str, Any]
    timestamp_ms: int = field(default_factory=now_ms)
    frame_id: str | None = None
    expires_at_ms: int | None = None
    geometry: dict[str, Any] = field(default_factory=dict)
    style: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeviceEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp_ms: int = field(default_factory=now_ms)

    def to_message(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeviceStatus:
    connected: bool
    streaming: bool
    timestamp_ms: int = field(default_factory=now_ms)
    battery_percent: float | None = None
    transport: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> dict[str, Any]:
        return asdict(self)
