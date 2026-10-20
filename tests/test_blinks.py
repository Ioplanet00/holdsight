import numpy as np
from holdsight.config import Thresholds
from holdsight.profiles.driver import blink_events, summarise

T = Thresholds()


def series(p5, fps=30, seed=0):
    rng = np.random.default_rng(seed)
    sig = [None if v is None else (v + rng.normal(0, 1.0), 0.3 - (v - 40) / 400 + rng.normal(0, 0.005)) for v in p5]
    return sig, [i * 1000.0 / fps for i in range(len(p5))]


def test_one_blink_detected():
    sig, ts = series([40] * 90 + [80] * 4 + [40] * 90)
    blinks, closures, _ = blink_events(sig, ts, T)
    assert len(blinks) == 1 and closures == []


def test_long_closure_is_not_a_blink():
    sig, ts = series([40] * 120 + [80] * 30 + [40] * 120)
    blinks, closures, _ = blink_events(sig, ts, T)
    assert blinks == [] and len(closures) == 1


def test_held_frames_are_never_closed():
    sig, ts = series([40] * 90 + [None] * 4 + [40] * 90)
    blinks, closures, _ = blink_events(sig, ts, T)
    assert blinks == [] and closures == []
    s = summarise(sig, ts, usable_s=6.0, t=T)
    assert s["blink_count"] == 0 and s["perclos"] == 0


def test_steady_noise_gives_no_blinks():
    sig, ts = series([40] * 600)
    assert blink_events(sig, ts, T)[0] == []
