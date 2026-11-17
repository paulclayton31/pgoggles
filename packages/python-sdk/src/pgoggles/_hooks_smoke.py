"""Quick smoke test for the new hooks module."""

from pgoggles.hooks import HookRegistry, HookHandle, DetectionContext
from pgoggles.overlays import BoundingBox, DetectionMode


def test_stale_mode():
    registry = HookRegistry()
    calls = []
    handle = registry.register(
        labels={"person"},
        mode=DetectionMode.STALE,
        callback=lambda boxes, ctx: calls.append((len(boxes), ctx.frame_id)),
    )
    assert registry.count == 1

    boxes = [BoundingBox(0, 0, 100, 100, "person", 0.9)]
    registry.evaluate(boxes, frame_id=1, timestamp=1000.0)
    assert len(calls) == 1, f"Expected 1 call, got {len(calls)}"
    print(f"  Frame 1: calls={calls}")

    # STALE should not fire again for same detection
    registry.evaluate(boxes, frame_id=2, timestamp=1001.0)
    assert len(calls) == 1, f"Expected still 1 call (STALE), got {len(calls)}"
    print(f"  Frame 2 (STALE no-fire): calls={calls}")

    # Detection disappears then reappears -> should fire again
    registry.evaluate([], frame_id=3, timestamp=1002.0)
    registry.evaluate(boxes, frame_id=4, timestamp=1003.0)
    assert len(calls) == 2, f"Expected 2 calls after disappear/reappear, got {len(calls)}"
    print(f"  Frame 4 (reappear): calls={calls}")

    handle.unregister()
    assert handle.unregistered is True
    assert registry.count == 0
    print("  stale mode: PASS")


def test_active_mode():
    registry = HookRegistry()
    calls = []
    handle = registry.register(
        labels={"car"},
        mode=DetectionMode.ACTIVE,
        debounce_ms=100,  # 0.1s
        callback=lambda boxes, ctx: calls.append((len(boxes), ctx.frame_id)),
    )

    car_boxes = [BoundingBox(0, 0, 50, 50, "car", 0.85)]
    registry.evaluate(car_boxes, frame_id=1, timestamp=1000.0)
    assert len(calls) == 1
    print(f"  Frame 1: calls={calls}")

    # Within debounce -> no fire
    registry.evaluate(car_boxes, frame_id=2, timestamp=1000.05)
    assert len(calls) == 1
    print(f"  Frame 2 (debounce): calls={calls}")

    # After debounce -> fires
    registry.evaluate(car_boxes, frame_id=3, timestamp=1000.2)
    assert len(calls) == 2
    print(f"  Frame 3 (after debounce): calls={calls}")

    handle.unregister()
    print("  active mode: PASS")


def test_cleanup():
    registry = HookRegistry()
    cleaned = []
    handle = registry.register(
        labels={"dog"},
        mode=DetectionMode.STALE,
        callback=lambda boxes, ctx: None,
        cleanup=lambda: cleaned.append(True),
    )
    assert len(cleaned) == 0
    handle.unregister()
    assert len(cleaned) == 1
    print("  cleanup: PASS")


def test_clear():
    registry = HookRegistry()
    cleaned = []
    registry.register(
        labels={"a"},
        mode=DetectionMode.STALE,
        callback=lambda boxes, ctx: None,
        cleanup=lambda: cleaned.append("a"),
    )
    registry.register(
        labels={"b"},
        mode=DetectionMode.STALE,
        callback=lambda boxes, ctx: None,
        cleanup=lambda: cleaned.append("b"),
    )
    assert registry.count == 2
    registry.clear()
    assert registry.count == 0
    assert sorted(cleaned) == ["a", "b"]
    print("  clear: PASS")


def test_multiple_hooks():
    registry = HookRegistry()
    person_calls = []
    car_calls = []

    registry.register(
        labels={"person"},
        mode=DetectionMode.ACTIVE,
        callback=lambda boxes, ctx: person_calls.append(ctx.frame_id),
    )
    registry.register(
        labels={"car"},
        mode=DetectionMode.ACTIVE,
        callback=lambda boxes, ctx: car_calls.append(ctx.frame_id),
    )

    boxes = [
        BoundingBox(0, 0, 100, 100, "person", 0.9),
        BoundingBox(50, 50, 150, 150, "car", 0.8),
    ]
    registry.evaluate(boxes, frame_id=1, timestamp=1000.0)
    assert len(person_calls) == 1
    assert len(car_calls) == 1
    print(f"  Both hooks fired: person={person_calls}, car={car_calls}")

    registry.clear()
    print("  multiple hooks: PASS")


if __name__ == "__main__":
    print("Running hooks smoke tests...")
    test_stale_mode()
    test_active_mode()
    test_cleanup()
    test_clear()
    test_multiple_hooks()
    print("All smoke tests passed!")
