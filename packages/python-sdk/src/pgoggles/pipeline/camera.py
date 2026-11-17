"""Threaded camera capture source for the pgoggles video pipeline."""

from __future__ import annotations

import logging
import queue
import threading
import time

import cv2

from ..overlays import FramePacket

logger = logging.getLogger(__name__)


class CameraSource:
    """Capture frames from a camera and publish the freshest packets.

    The ESP32-S3 OV2640 transport should replace this class behind the same
    start/stop/latest_frame/queue boundary, leaving analysis and display intact.
    """

    def __init__(self, camera_index: int = 0, queue_size: int = 2) -> None:
        # TODO: Replace cv2.VideoCapture with an ESP32 transport-backed source.
        self.camera_index = camera_index
        self.frame_queue: queue.Queue[FramePacket] = queue.Queue(maxsize=queue_size)
        self._capture: cv2.VideoCapture | None = None
        self._frame_id = 0
        self._latest_frame: FramePacket | None = None
        self._latest_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def latest_frame(self) -> FramePacket | None:
        """Return the freshest captured frame packet without consuming history."""

        with self._latest_lock:
            return self._latest_frame

    def start(self) -> None:
        """Start the capture thread."""

        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="pgoggles-camera",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop capture and release the camera device."""

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _capture_loop(self) -> None:
        self._capture = cv2.VideoCapture(self.camera_index)
        if not self._capture.isOpened():
            logger.warning("Could not open camera index %s", self.camera_index)
            return

        while not self._stop_event.is_set():
            ok, frame = self._capture.read()
            if not ok:
                logger.warning("Camera frame read failed")
                time.sleep(0.05)
                continue

            self._frame_id += 1
            packet = FramePacket(
                frame_id=self._frame_id,
                timestamp=time.time(),
                frame=frame,
            )
            with self._latest_lock:
                self._latest_frame = packet
            self._push_latest(packet)

    def _push_latest(self, packet: FramePacket) -> None:
        try:
            self.frame_queue.put_nowait(packet)
            return
        except queue.Full:
            pass

        try:
            self.frame_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.frame_queue.put_nowait(packet)
        except queue.Full:
            logger.debug("Dropped frame %s because capture queue stayed full", packet.frame_id)
