"""Hold reason codes. Every frame/session is either a value or exactly one hold code."""
from enum import Enum


class Hold(str, Enum):
    NO_FACE = "NO_FACE"        # no face detected or score below threshold
    NO_EYE = "NO_EYE"          # no valid eye contour, or the eye box is covered (texture collapse)
    GLARE = "GLARE"            # saturated pixels / largest saturated blob in eye box above threshold
    LOW_LIGHT = "LOW_LIGHT"    # face-box mean intensity below threshold
    MOTION = "MOTION"          # face-box displacement above threshold, or blur
    LOW_FPS = "LOW_FPS"        # session: delivered fps below minimum
    LOW_USABLE = "LOW_USABLE"  # session: usable-frame share below minimum or too short


WHY = {
    Hold.NO_FACE: "No face was found in this part of the video, so nothing was measured.",
    Hold.NO_EYE: "The face was found but the eye region could not be read (for example it was covered).",
    Hold.GLARE: "Strong reflection (for example from glasses) covered the eye.",
    Hold.LOW_LIGHT: "The eye region was too dark to measure reliably.",
    Hold.MOTION: "The head moved too fast or the image was blurred.",
    Hold.LOW_FPS: "The video frame rate was too low for blink timing.",
    Hold.LOW_USABLE: "Too few frames could be measured, so no session result is given.",
}
# Frame-level precedence when several checks fail at once (first wins).
FRAME_PRECEDENCE = [Hold.NO_FACE, Hold.LOW_LIGHT, Hold.GLARE, Hold.NO_EYE, Hold.MOTION]
