from holdsight.config import Thresholds
from holdsight.profiles.driver import blink_events, summarise

T = Thresholds()


def series(pattern, fps=30):
    return pattern, [i * 1000.0 / fps for i in range(len(pattern))]


def test_one_blink_detected():
    ap, ts = series([0.5] * 30 + [0.1] * 4 + [0.5] * 30)
    blinks, closures, _ = blink_events(ap, ts, T)
    assert len(blinks) == 1 and closures == []


def test_long_closure_is_not_a_blink():
    ap, ts = series([0.5] * 30 + [0.1] * 30 + [0.5] * 30)
    blinks, closures, _ = blink_events(ap, ts, T)
    assert blinks == [] and len(closures) == 1


def test_held_frames_break_events_and_are_not_closed():
    ap, ts = series([0.5] * 30 + [0.1, None, None, 0.1] + [0.5] * 30)
    blinks, _, _ = blink_events(ap, ts, T)
    assert blinks == []
    s = summarise(ap, ts, usable_s=2.0, t=T)
    assert s["blink_count"] == 0


def test_single_frame_dip_ignored():
    ap, ts = series([0.5] * 30 + [0.1] + [0.5] * 30)
    assert blink_events(ap, ts, T)[0] == []
