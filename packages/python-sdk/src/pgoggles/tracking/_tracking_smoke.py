"""Quick smoke test for the tracking module (IoU, assignment, Kalman, tracker).

Pure, synthetic-box math -- no camera/model/DB needed. Mirrors the style of
``pgoggles/_hooks_smoke.py`` (bare asserts, printed pass markers, plain
``if __name__ == "__main__"`` runner). Run with:

    python -m pgoggles.tracking._tracking_smoke
"""

import numpy as np

from pgoggles.overlays import BoundingBox, TrackingMode
from pgoggles.tracking.assignment import greedy_match, hungarian_match
from pgoggles.tracking.iou import iou, iou_matrix
from pgoggles.tracking.kalman import KalmanBoxTracker
from pgoggles.tracking.oc_sort import (
    augmented_cost_matrix,
    direction_consistency_matrix,
    interpolate_boxes,
    reupdate_kalman,
)
from pgoggles.tracking.tracker import MultiObjectTracker


def test_iou_perfect_overlap():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    print("  iou perfect overlap: PASS")


def test_iou_no_overlap():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    print("  iou no overlap: PASS")


def test_iou_partial_overlap():
    # Two 10x10 boxes overlapping in a 5x10 strip: intersection=50, union=150
    value = iou((0, 0, 10, 10), (5, 0, 15, 10))
    expected = 50 / 150
    assert abs(value - expected) < 1e-9, f"expected {expected}, got {value}"
    print(f"  iou partial overlap: PASS ({value:.4f})")


def test_greedy_match_trivial():
    # 1x1
    matches, unmatched_t, unmatched_d = greedy_match(np.array([[0.6]]), 0.3)
    assert matches == [(0, 0)] and not unmatched_t and not unmatched_d

    # 0xN / Nx0 edge cases
    matches, unmatched_t, unmatched_d = greedy_match(np.zeros((0, 3)), 0.3)
    assert matches == [] and unmatched_t == [] and unmatched_d == [0, 1, 2]
    matches, unmatched_t, unmatched_d = greedy_match(np.zeros((3, 0)), 0.3)
    assert matches == [] and unmatched_t == [0, 1, 2] and unmatched_d == []
    print("  greedy match trivial cases: PASS")


def test_hungarian_match_optimal_assignment():
    # Classic case where greedy's locally-best-first pick is suboptimal:
    # greedy takes (0,0)=0.5 then is forced into (1,1)=0.3 -> total 0.8
    # Hungarian finds (0,1)=0.4 + (1,0)=0.45 -> total 0.85
    iou_mat = np.array([[0.5, 0.4], [0.45, 0.3]])

    def total(matches):
        return sum(iou_mat[t, d] for t, d in matches)

    greedy_matches, _, _ = greedy_match(iou_mat, 0.1)
    hungarian_matches, _, _ = hungarian_match(iou_mat, 0.1)
    assert total(hungarian_matches) > total(greedy_matches), (
        f"expected hungarian ({total(hungarian_matches)}) > greedy ({total(greedy_matches)})"
    )
    print(f"  hungarian optimal assignment: PASS (greedy={total(greedy_matches)}, hungarian={total(hungarian_matches)})")


def test_hungarian_short_circuits_to_greedy_when_trivial():
    import sys

    # Simulate scipy being unavailable; the (<=1, <=1) short-circuit in
    # hungarian_match must never attempt the import in this shape.
    saved = {name: sys.modules.get(name) for name in ("scipy", "scipy.optimize")}
    sys.modules["scipy"] = None
    sys.modules["scipy.optimize"] = None
    try:
        trivial = np.array([[0.6]])
        assert hungarian_match(trivial, 0.3) == greedy_match(trivial, 0.3)
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
    print("  hungarian trivial short-circuit (no scipy import): PASS")


def test_kalman_predict_converges():
    from pgoggles.tracking.kalman import KalmanBoxTracker

    kf = KalmanBoxTracker(bbox=(0, 0, 20, 20))
    for step in range(1, 6):
        kf.predict()
        kf.update((5 * step, 0, 20 + 5 * step, 20))

    predicted = kf.predict()
    expected = (30, 0, 50, 20)
    for got, want in zip(predicted, expected):
        assert abs(got - want) < 1.0, f"predicted {predicted} too far from expected {expected}"
    print(f"  kalman predict converges: PASS (predicted={tuple(round(v, 2) for v in predicted)})")


def test_track_lifecycle_confirm_and_expire():
    tracker = MultiObjectTracker(mode=TrackingMode.FULL, min_hits=3, max_age=2)

    box = lambda x: [BoundingBox(x1=x, y1=0, x2=x + 10, y2=10, label="person", confidence=0.9)]

    # Frames 0-1: tentative, not yet reported
    assert tracker.update(box(0)) == []
    assert tracker.update(box(1)) == []
    # Frame 2: hits reaches min_hits=3 -> confirmed and reported
    out = tracker.update(box(2))
    assert len(out) == 1 and out[0].track_id == 1 and out[0].confirmed
    track_id = out[0].track_id

    # Detection stops appearing -- track coasts for max_age=2 frames, then is pruned
    out = tracker.update([])
    assert len(out) == 1 and out[0].track_id == track_id and out[0].time_since_update == 1
    out = tracker.update([])
    assert len(out) == 1 and out[0].track_id == track_id and out[0].time_since_update == 2
    out = tracker.update([])
    assert out == [], f"expected track pruned after max_age, got {out}"
    print("  track lifecycle confirm+expire: PASS")


def test_disabled_mode_is_passthrough():
    tracker = MultiObjectTracker(mode=TrackingMode.DISABLED)
    boxes = [BoundingBox(x1=0, y1=0, x2=10, y2=10, label="person", confidence=0.9)]
    out = tracker.update(boxes)
    assert len(out) == 1 and out[0].track_id == -1
    assert tracker.tracks == [], "DISABLED mode must create no internal Track state"
    print("  disabled mode passthrough: PASS")


def test_per_label_bucketing_no_cross_class_match():
    tracker = MultiObjectTracker(mode=TrackingMode.FULL, min_hits=1)

    # A "car" and "person" box with identical (fully-overlapping) geometry --
    # spatially they'd be a perfect IoU=1.0 match, but must never share a track.
    boxes = [
        BoundingBox(x1=0, y1=0, x2=20, y2=20, label="person", confidence=0.9),
        BoundingBox(x1=0, y1=0, x2=20, y2=20, label="car", confidence=0.9),
    ]
    out = tracker.update(boxes)
    ids_by_label = {b.label: b.track_id for b in out}
    assert ids_by_label["person"] != ids_by_label["car"], "cross-class boxes must not share a track_id"

    # Confirm stability: same labels, same geometry next frame -> same ids
    out2 = tracker.update(boxes)
    ids_by_label2 = {b.label: b.track_id for b in out2}
    assert ids_by_label == ids_by_label2
    print("  per-label bucketing (no cross-class match): PASS")


def test_coasting_track_reported_while_unmatched():
    tracker = MultiObjectTracker(mode=TrackingMode.KALMAN_IOU, min_hits=1, max_age=3)
    box = [BoundingBox(x1=0, y1=0, x2=10, y2=10, label="person", confidence=0.9)]

    out = tracker.update(box)
    assert len(out) == 1 and out[0].time_since_update == 0

    # Detection missed this frame -- track should still be reported (coasting),
    # using its predicted position, with increasing time_since_update.
    out = tracker.update([])
    assert len(out) == 1 and out[0].time_since_update == 1, "expected coasting track in output"
    out = tracker.update([])
    assert len(out) == 1 and out[0].time_since_update == 2
    print("  coasting track reported while unmatched: PASS")


def test_direction_consistency_rewards_aligned_motion():
    history = [[(0, 0, 10, 10), (5, 0, 15, 10), (10, 0, 20, 10)]]  # moving rightward
    continuing = (15, 0, 25, 10)  # continues rightward
    reversing = (5, 0, 15, 10)  # reverses back toward where it came from

    mat = direction_consistency_matrix(history, [continuing, reversing], delta_t=3)
    assert mat[0, 0] > mat[0, 1], f"expected continuation to score higher than reversal: {mat}"

    # Fewer than 2 real observations -> neutral (no established direction).
    neutral = direction_consistency_matrix([[(0, 0, 10, 10)]], [continuing], delta_t=3)
    assert (neutral == 0.0).all(), neutral

    print(f"  direction consistency rewards aligned motion: PASS ({mat[0, 0]:.2f} vs {mat[0, 1]:.2f})")


def test_ocm_gate_blocks_zero_iou_pairs():
    history = [[(0, 0, 10, 10), (5, 0, 15, 10), (10, 0, 20, 10)]]
    detections = [(15, 0, 25, 10)]  # perfectly aligned motion continuation

    zero_iou = np.array([[0.0]])
    augmented = augmented_cost_matrix(zero_iou, history, detections, ocm_weight=0.2, delta_t=3)
    assert augmented[0, 0] == 0.0, (
        f"momentum must never turn a zero-IoU pair into a nonzero score, got {augmented[0, 0]}"
    )

    nonzero_iou = np.array([[0.25]])
    boosted = augmented_cost_matrix(nonzero_iou, history, detections, ocm_weight=0.2, delta_t=3)
    assert boosted[0, 0] > 0.25, f"expected aligned momentum to boost a real-overlap pair: {boosted}"

    print("  ocm gate blocks zero-iou pairs: PASS")


def test_interpolate_boxes_linear_and_exact_endpoint():
    boxes = interpolate_boxes((0, 0, 10, 10), (30, 0, 40, 10), steps=3)
    assert boxes == [
        (10.0, 0.0, 20.0, 10.0),
        (20.0, 0.0, 30.0, 10.0),
        (30.0, 0.0, 40.0, 10.0),
    ], boxes
    assert interpolate_boxes((0, 0, 10, 10), (5, 5, 15, 15), steps=1) == [(5.0, 5.0, 15.0, 15.0)]

    try:
        interpolate_boxes((0, 0, 10, 10), (5, 5, 15, 15), steps=0)
        raise AssertionError("expected ValueError for steps=0")
    except ValueError:
        pass

    print("  interpolate_boxes linear and exact endpoint: PASS")


def test_oru_produces_sane_velocity_not_corrupted_drift():
    def _establish(kf: KalmanBoxTracker) -> None:
        for step in range(1, 4):
            kf.predict()
            kf.update((10 * step, 0, 20 + 10 * step, 20))  # steady rightward motion, vx -> ~10

    kf_baseline = KalmanBoxTracker(bbox=(0, 0, 20, 20))
    _establish(kf_baseline)

    last_observation = (30, 0, 50, 20)  # position after the 3 setup steps
    new_observation = (30, 0, 50, 20)  # object then stops for a 3-frame gap -- true vx during the gap was 0

    # Baseline: today's pre-ORU coasting -- 3 blind predict()-only steps, then one correction.
    for _ in range(3):
        kf_baseline.predict()
    kf_baseline.update(new_observation)

    # ORU: rebuild from scratch via straight-line replay across the same gap.
    kf_reupdated = reupdate_kalman(last_observation, new_observation, steps=3)

    baseline_vx = abs(kf_baseline._x[4])
    reupdate_vx = abs(kf_reupdated._x[4])
    assert reupdate_vx < baseline_vx, (
        f"expected ORU's vx ({reupdate_vx}) closer to the true value (0) than blind coasting's ({baseline_vx})"
    )
    print(f"  oru produces sane velocity: PASS (baseline_vx={baseline_vx:.3f}, reupdate_vx={reupdate_vx:.3f})")


def test_ocr_recovers_sudden_stop_primary_match_would_miss():
    def box(x: int) -> BoundingBox:
        return BoundingBox(x1=x, y1=0, x2=x + 40, y2=10, label="person", confidence=0.9)

    with_ocr = MultiObjectTracker(mode=TrackingMode.FULL, min_hits=1, iou_threshold=0.3, use_ocr=True)
    without_ocr = MultiObjectTracker(mode=TrackingMode.FULL, min_hits=1, iou_threshold=0.3, use_ocr=False)

    # Establish steady vx=10 over several well-overlapping frames.
    for x in (0, 10, 20, 30):
        with_ocr.update([box(x)])
        without_ocr.update([box(x)])

    # Occluded for 2 frames -- object has actually stopped, but nothing is detected.
    with_ocr.update([])
    without_ocr.update([])
    with_ocr.update([])
    without_ocr.update([])

    # Reappears stopped at the same position it was last really observed at.
    # The Kalman-predicted box has overshot well past x=30 by now, so primary
    # (predicted-box) matching alone should miss this.
    out_with = with_ocr.update([box(30)])
    out_without = without_ocr.update([box(30)])

    assert len(out_with) == 1 and out_with[0].track_id == 1 and out_with[0].time_since_update == 0, out_with

    same_id_matched = any(b.track_id == 1 and b.time_since_update == 0 for b in out_without)
    assert not same_id_matched, f"expected OCR-off to fail to cleanly recover: {out_without}"

    print("  ocr recovers sudden-stop-after-occlusion match: PASS")


def test_index_remapping_no_double_counting_in_ocr():
    def box(x: int, y: int) -> BoundingBox:
        return BoundingBox(x1=x, y1=y, x2=x + 40, y2=y + 10, label="person", confidence=0.9)

    tracker = MultiObjectTracker(mode=TrackingMode.FULL, min_hits=1, iou_threshold=0.3)

    # Three parallel lanes, well-separated.
    for i in range(4):
        tracker.update([box(10 * i, 0), box(200 + 10 * i, 100), box(400 + 10 * i, 200)])
    tracker.update([box(30, 0), box(230, 100), box(430, 200)])

    id_lane0, id_lane1, id_lane2 = [
        t.track_id for t in sorted(tracker.tracks, key=lambda t: t.last_observation[1])
    ]

    # Occlude lanes 0 and 2 for two frames; lane 1 keeps moving and matching normally.
    tracker.update([box(240, 100)])
    tracker.update([box(250, 100)])

    # Lanes 0 and 2 reappear stopped at their last real positions; lane 1 continues moving.
    out = tracker.update([box(30, 0), box(260, 100), box(430, 200)])

    ids_out = [b.track_id for b in out]
    assert len(ids_out) == len(set(ids_out)), f"duplicate track_id in output: {ids_out}"

    by_id = {b.track_id: b for b in out}
    assert by_id[id_lane0].y1 == 0 and by_id[id_lane0].time_since_update == 0
    assert by_id[id_lane2].y1 == 200 and by_id[id_lane2].time_since_update == 0
    assert by_id[id_lane1].y1 == 100 and by_id[id_lane1].time_since_update == 0

    print("  index remapping no double counting in ocr: PASS")


if __name__ == "__main__":
    print("Running tracking smoke tests...")
    test_iou_perfect_overlap()
    test_iou_no_overlap()
    test_iou_partial_overlap()
    test_greedy_match_trivial()
    test_hungarian_match_optimal_assignment()
    test_hungarian_short_circuits_to_greedy_when_trivial()
    test_kalman_predict_converges()
    test_track_lifecycle_confirm_and_expire()
    test_disabled_mode_is_passthrough()
    test_per_label_bucketing_no_cross_class_match()
    test_coasting_track_reported_while_unmatched()
    test_direction_consistency_rewards_aligned_motion()
    test_ocm_gate_blocks_zero_iou_pairs()
    test_interpolate_boxes_linear_and_exact_endpoint()
    test_oru_produces_sane_velocity_not_corrupted_drift()
    test_ocr_recovers_sudden_stop_primary_match_would_miss()
    test_index_remapping_no_double_counting_in_ocr()
    print("All smoke tests passed!")
