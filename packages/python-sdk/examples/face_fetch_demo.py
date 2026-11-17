"""Run the pgoggles face-fetch demo.

This demo wires the pgoggles SDK's onDetect pipeline into the
``research/identity`` package (embedding + pgvector database lookup) to
answer "who is this person" live from the camera feed:

1. **onDetect("person")** — an ACTIVE-mode hook fires every frame a person
   is detected, exposing the current frame crop to an embedding consumer
   thread.
2. **Embedding consumer thread** — a background worker turns the frame crop
   into a 512-dim facial embedding (``FaceEmbedder``) and uses it as a key
   into the ``research.identity`` Postgres+pgvector database to fetch the
   matching person's name (``find_nearest``).
3. **Draw command** — once the lookup resolves, a ``DrawBoundingBox``
   command labelled with the person's name is enqueued into the draw
   pipeline and kept up to date for as long as the person remains detected.
4. **onDetect("person"), GONE mode** — fires once the person is no longer
   detected, clearing the resolved identity and removing the draw command.

Requires the identity research package's dependencies and a configured
database — see ``research/identity/README.md``:

    PGOGGLES_INSTALL_IDENTITY_RESEARCH=1 ./setup.sh
    PGOGGLES_IDENTITY_RESEARCH_ENABLED=1 PGOGGLES_DATABASE_URL=... \\
        python packages/python-sdk/examples/face_fetch_demo.py

See the module docstring at the bottom of this file for a note on the one
remaining SDK gap this demo works around (no persistent per-detection
tracking across frames).

Usage::

    python examples/face_fetch_demo.py
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # packages/python-sdk
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REPO_ROOT = Path(__file__).resolve().parents[3]  # pgoggles/ (repo root)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pgoggles.hooks import DetectionContext
from pgoggles.overlays import BoundingBox, DetectionMode, DrawBoundingBox
from pgoggles.pipeline import AnalysisPipeline, CameraSource, DrawPipeline
from pgoggles.display import DisplayWindow

CONFIG = {
    "CAMERA_INDEX": 0,
    "ANALYSIS_HZ_MAX": 10.0,
    "SCENE_CHANGE_THRESHOLD": 8.0,
    # Export once in an environment with ultralytics: yolo export model=yolov8n.pt format=onnx
    # Hardcoded relative to ROOT (packages/python-sdk/) rather than left as a bare
    # filename, since AnalysisPipeline resolves YOLO_MODEL against the process's
    # current working directory, not the script's location.
    "YOLO_MODEL": str(ROOT / "yolov8n.onnx"),
    "DISPLAY_FPS_TARGET": 30.0,
    "PERSON_CONFIDENCE_THRESHOLD": 0.6,
}

_PERSON_COMMAND_ID = "face_fetch_person"


class _PersonIdentity:
    """Resolution state for "the current person in frame".

    pgoggles has no persistent per-detection track IDs yet (bounding boxes
    carry no identity across frames), so this demo can only track a single
    "current person" slot rather than distinguishing multiple simultaneous
    faces. See the module docstring at the bottom of this file for details.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.name: str | None = None
        self.pending = False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Load .env before checking the opt-in flag below. research/identity/db/database.py
    # also calls load_dotenv(), but only *after* the flag check passes below — by then
    # it's too late for a flag that's set in .env rather than exported in the shell.
    # python-dotenv itself is one of the optional identity-research dependencies, so
    # tolerate it not being installed yet (PGOGGLES_IDENTITY_RESEARCH_ENABLED must then
    # be exported directly, e.g. `PGOGGLES_IDENTITY_RESEARCH_ENABLED=1 python ...`).
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if os.getenv("PGOGGLES_IDENTITY_RESEARCH_ENABLED") != "1":
        raise SystemExit(
            "Identity research is disabled. Set "
            "PGOGGLES_IDENTITY_RESEARCH_ENABLED=1 to run this demo."
        )

    # Imported lazily, behind the opt-in check above, since these pull in
    # the optional identity-research dependencies (insightface, sqlalchemy, ...).
    from research.identity.db.database import get_session
    from research.identity.db.queries import find_nearest
    from research.identity.embeddings.face_embedder import FaceEmbedder

    # 1. Start the camera source
    camera = CameraSource(camera_index=CONFIG["CAMERA_INDEX"])

    # 2. Create the draw pipeline (independent threaded stage)
    draw_pipeline = DrawPipeline()

    # 3. Start the analysis pipeline (YOLO inference)
    analysis = AnalysisPipeline(
        camera,
        hz_max=CONFIG["ANALYSIS_HZ_MAX"],
        scene_change_threshold=CONFIG["SCENE_CHANGE_THRESHOLD"],
        yolo_model=CONFIG["YOLO_MODEL"],
        draw_pipeline=draw_pipeline,
    )

    # 4. Configure the display window (independent threaded stage)
    display_window = DisplayWindow(
        camera_source=camera,
        analysis_pipeline=analysis,
        draw_pipeline=draw_pipeline,
        fps_target=CONFIG["DISPLAY_FPS_TARGET"],
        window_name="pgoggles face-fetch demo — person appears, fetches name from db",
    )

    embedder = FaceEmbedder()
    session = get_session()

    identity = _PersonIdentity()
    embed_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    # ------------------------------------------------------------------
    # 2. Embedding consumer thread: frame crop -> embedding -> db lookup
    # ------------------------------------------------------------------
    def _embedding_worker() -> None:
        while not stop_event.is_set():
            try:
                crop = embed_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                embedding = embedder.embed(crop)
                name: str | None
                if embedding is None:
                    name = None
                else:
                    matches = find_nearest(session, embedding, limit=1)
                    name = matches[0].name if matches else "Unknown"
            except Exception:
                logging.exception("face-fetch: embedding/lookup failed")
                name = None

            with identity.lock:
                identity.pending = False
                if name is not None:
                    identity.name = name

    worker_thread = threading.Thread(
        target=_embedding_worker, name="pgoggles-face-fetch-embedder", daemon=True
    )
    worker_thread.start()

    # ------------------------------------------------------------------
    # 1. onDetect hook: expose the person crop, draw whatever we know so far
    # ------------------------------------------------------------------
    def _on_person_active(
        boxes: list[BoundingBox],
        context: DetectionContext,
        draw_pipeline: DrawPipeline,
    ) -> None:
        best = max(boxes, key=lambda b: b.confidence)

        with identity.lock:
            name = identity.name
            should_submit = (
                name is None and not identity.pending and context.frame is not None
            )
            if should_submit:
                identity.pending = True

        if should_submit:
            crop = context.frame[best.y1 : best.y2, best.x1 : best.x2].copy()
            try:
                embed_queue.put_nowait(crop)
            except queue.Full:
                with identity.lock:
                    identity.pending = False

        label = name or "Identifying..."
        draw_pipeline.enqueue(
            DrawBoundingBox(
                x1=best.x1,
                y1=best.y1,
                x2=best.x2,
                y2=best.y2,
                label=label,
                confidence=best.confidence,
                command_id=_PERSON_COMMAND_ID,
            )
        )

    analysis.hooks.register(
        labels={"person"},
        mode=DetectionMode.ACTIVE,
        confidence_threshold=CONFIG["PERSON_CONFIDENCE_THRESHOLD"],
        debounce_ms=0.0,
        callback=_on_person_active,
    )

    # ------------------------------------------------------------------
    # 4. onDetect hook (GONE): person left the frame — clear the resolved
    # identity and remove the draw command. Replaces the watchdog-thread
    # timeout workaround now that HookRegistry supports a disappearance
    # trigger natively.
    # ------------------------------------------------------------------
    def _on_person_gone(
        boxes: list[BoundingBox],
        context: DetectionContext,
        draw_pipeline: DrawPipeline,
    ) -> None:
        with identity.lock:
            identity.name = None
            identity.pending = False
        logging.info("face-fetch: person gone, clearing identity and draw command")
        draw_pipeline.remove_command(_PERSON_COMMAND_ID)

    analysis.hooks.register(
        labels={"person"},
        mode=DetectionMode.GONE,
        confidence_threshold=CONFIG["PERSON_CONFIDENCE_THRESHOLD"],
        debounce_ms=0.0,
        callback=_on_person_gone,
    )

    # 5. Start all pipeline stages
    camera.start()
    draw_pipeline.start()
    analysis.start()

    try:
        # 6. Run the display window (blocking until the user presses "q").
        display_window.run()
    finally:
        stop_event.set()
        worker_thread.join(timeout=2.0)
        session.close()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# SDK sufficiency notes (see task report)
# ---------------------------------------------------------------------------
#
# The pgoggles SDK, as it stands, is sufficient to implement this demo. One
# gap remains, to be addressed next as a tracking/id-persistence project:
#
# No detection identity / tracking. BoundingBox carries no persistent ID
# across frames, so there's no way to distinguish "person A" from "person B"
# if two people are ever in frame together, or to know that the person seen
# this frame is the "same" one seen last frame beyond proximity in time.
# This demo treats "the current person" as a single slot, which is correct
# for the single-person case the task describes but would need a tracker
# (e.g. IoU-based or embedding-based re-ID) to generalize to multiple
# simultaneous people.
#
# (Resolved) No "detection ended" hook — HookRegistry now has
# DetectionMode.GONE, which fires once when a previously-present detection
# disappears, handing the callback the last-seen matching boxes. See
# pgoggles/hooks.py::HookRegistry.evaluate.
