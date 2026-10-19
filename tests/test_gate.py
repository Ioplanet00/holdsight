from holdsight.codes import Hold
from holdsight.config import Thresholds
from holdsight.eye import EyeReading
from holdsight.gate import frame_hold, session_hold

T = Thresholds()
GOOD = EyeReading(aperture=0.5, pupil=(0.5, 0.5), mean=120, glare_ratio=0.0, blur_var=200)


def test_usable_frame_has_no_hold():
    assert frame_hold(True, 0.0, GOOD, T) is None


def test_no_face_wins():
    assert frame_hold(False, None, None, T) == Hold.NO_FACE


def test_each_frame_code():
    assert frame_hold(True, 1.0, GOOD, T) == Hold.MOTION
    assert frame_hold(True, 0.0, EyeReading(0.5, None, 10, 0.0, 200), T) == Hold.LOW_LIGHT
    assert frame_hold(True, 0.0, EyeReading(0.5, None, 120, 0.5, 200), T) == Hold.GLARE
    assert frame_hold(True, 0.0, EyeReading(None, None, 120, 0.0, 200), T) == Hold.NO_EYE
    assert frame_hold(True, 0.0, EyeReading(0.5, None, 120, 0.0, 1), T) == Hold.MOTION


def test_session_codes():
    assert session_hold(8, 1.0, 60, T) == Hold.LOW_FPS
    assert session_hold(30, 0.2, 60, T) == Hold.LOW_USABLE
    assert session_hold(30, 1.0, 3, T) == Hold.LOW_USABLE
    assert session_hold(30, 1.0, 60, T) is None
