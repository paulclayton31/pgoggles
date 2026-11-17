"""Async plugin lifecycle for consuming frames and emitting overlays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .device import Device
from .messages import DeviceEvent, Frame, Overlay, Pose


@dataclass
class PluginContext:
    device: Device
    state: dict[str, Any] = field(default_factory=dict)

    def emit(self, overlay: Overlay) -> None:
        self.device.render_overlay(overlay)


class Plugin(Protocol):
    async def setup(self, context: PluginContext) -> None: ...

    async def on_frame(self, frame: Frame, context: PluginContext) -> None: ...

    async def on_pose(self, pose: Pose, context: PluginContext) -> None: ...

    async def on_event(self, event: DeviceEvent, context: PluginContext) -> None: ...

    async def teardown(self, context: PluginContext) -> None: ...


class BasePlugin:
    async def setup(self, context: PluginContext) -> None:
        pass

    async def on_frame(self, frame: Frame, context: PluginContext) -> None:
        pass

    async def on_pose(self, pose: Pose, context: PluginContext) -> None:
        pass

    async def on_event(self, event: DeviceEvent, context: PluginContext) -> None:
        pass

    async def teardown(self, context: PluginContext) -> None:
        pass


class PluginRuntime:
    def __init__(self, device: Device, plugins: list[Plugin] | None = None) -> None:
        self.context = PluginContext(device=device)
        self.plugins = plugins or []

    async def setup(self) -> None:
        for plugin in self.plugins:
            await plugin.setup(self.context)

    async def process_frame(self, frame: Frame) -> None:
        for plugin in self.plugins:
            await plugin.on_frame(frame, self.context)

    async def process_pose(self, pose: Pose) -> None:
        for plugin in self.plugins:
            await plugin.on_pose(pose, self.context)

    async def process_event(self, event: DeviceEvent) -> None:
        for plugin in self.plugins:
            await plugin.on_event(event, self.context)

    async def teardown(self) -> None:
        for plugin in reversed(self.plugins):
            await plugin.teardown(self.context)
