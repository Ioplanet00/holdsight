"""Smoke test on a synthetic face-free clip: everything must be held, nothing guessed."""
import os
import cv2 as cv
import numpy as np
import pytest
from holdsight.face import DEFAULT_MODEL


@pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="run make models")
def test_blank_video_is_held(tmp_path):
    from holdsight.pipeline import analyse
    p = str(tmp_path / "blank.mp4")
    vw = cv.VideoWriter(p, cv.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(60):
        vw.write(np.full((240, 320, 3), 128, np.uint8))
    vw.release()
    r = analyse(p)
    assert r["frame_holds"]["NO_FACE"] == r["frames"] == 60
    assert "hold" in r["session"] and "value" not in r["session"]
