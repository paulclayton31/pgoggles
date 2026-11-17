"""Transport placeholders for future hardware-specific implementations."""

from __future__ import annotations

from typing import Any, Protocol

from .device import TransportUnavailableError


class Transport(Protocol):
    name: str

    def connect(self) -> None: ...

    def close(self) -> None: ...

    def send(self, message: dict[str, Any]) -> None: ...

    def receive(self) -> dict[str, Any] | None: ...


class PlaceholderTransport:
    def __init__(self, name: str) -> None:
        self.name = name

    def connect(self) -> None:
        raise TransportUnavailableError(
            f"{self.name} transport is a skeleton placeholder until hardware details are known"
        )

    def close(self) -> None:
        pass

    def send(self, message: dict[str, Any]) -> None:
        raise TransportUnavailableError(f"{self.name} transport is not implemented")

    def receive(self) -> dict[str, Any] | None:
        raise TransportUnavailableError(f"{self.name} transport is not implemented")


def UsbTransport() -> PlaceholderTransport:
    return PlaceholderTransport("usb")


def BleTransport() -> PlaceholderTransport:
    return PlaceholderTransport("ble")


def WifiTransport() -> PlaceholderTransport:
    return PlaceholderTransport("wifi")
