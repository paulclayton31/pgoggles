"""Python SDK surface for pgoggles."""

from .device import Device, DisconnectedDeviceError, SimulatorDevice, StreamSample
from .display import DisplayWindow
from .messages import (
    DeviceEvent,
    DeviceStatus,
    EventType,
    Frame,
    FrameFormat,
    Overlay,
    OverlayType,
    Pose,
)
from .hooks import DetectionContext, HookHandle, HookRegistry
from .overlays import (
    ArrowDirection,
    BoundingBox,
    DetectionMode,
    DrawMode,
    DrawBoundingBox,
    DrawChevron,
    DrawCommand,
    DrawReticle,
    DrawText,
    ExposedFrame,
    FramePacket,
    OverlayState,
    TrackedBox,
    TrackingMode,
    bounding_box,
    chevron,
    notification,
    reticle,
    text,
)
from .plugins import BasePlugin, PluginContext, PluginRuntime
from .pipeline import AnalysisPipeline, CameraSource, DrawPipeline
from .tracking import MultiObjectTracker, Track
from .transports import BleTransport, Transport, TransportUnavailableError, UsbTransport, WifiTransport

__all__ = [
    # Device layer
    "AnalysisPipeline",
    "BasePlugin",
    "BleTransport",
    "ArrowDirection",
    "BoundingBox",
    "CameraSource",
    "DrawBoundingBox",
    "DrawPipeline",
    "DrawChevron",
    "DrawCommand",
    "DrawReticle",
    "DrawText",
    "DetectionContext",
    "DetectionMode",
    "ExposedFrame",
    "Device",
    "DeviceEvent",
    "DeviceStatus",
    "DisconnectedDeviceError",
    "DisplayWindow",
    "EventType",
    "Frame",
    "FrameFormat",
    "FramePacket",
    "HookHandle",
    "HookRegistry",
    "MultiObjectTracker",
    "Overlay",
    "OverlayState",
    "OverlayType",
    "PluginContext",
    "PluginRuntime",
    "Pose",
    "SimulatorDevice",
    "StreamSample",
    "Track",
    "TrackedBox",
    "TrackingMode",
    "Transport",
    "TransportUnavailableError",
    "UsbTransport",
    "WifiTransport",
    # Overlay constructors
    "bounding_box",
    "chevron",
    "notification",
    "reticle",
    "text",
]
