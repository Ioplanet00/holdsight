from dataclasses import dataclass, fields
from pathlib import Path
import yaml


@dataclass(frozen=True)
class Thresholds:
    face_score_min: float = 0.80
    eye_box_scale: float = 0.45          # eye box side = scale * inter-ocular distance
    eye_resize: int = 64                 # eye crop resized to eye_resize x eye_resize
    clahe_clip: float = 2.0
    clahe_tile: int = 4
    adapt_block: int = 15                # odd
    adapt_c: float = 5.0
    min_eye_contour_area: float = 20.0
    glare_level: int = 250
    glare_ratio_max: float = 0.08
    low_light_mean_min: float = 40.0
    motion_max_iod: float = 0.25         # face-centre shift per frame / inter-ocular distance
    blur_var_min: float = 15.0
    glare_blob_max: float = 0.012        # largest saturated blob / eye box area
    low_light_face_mean_min: float = 40.0  # face-box mean grey level
    occlusion_texture_min: float = 8.0   # raw eye-box std below this = covered eye
    occlusion_flat_max: float = 0.50     # share of near-zero-gradient pixels above this = covered eye
    blink_window_s: float = 2.0          # rolling baseline window for the eye-darkness signals
    blink_z: float = 3.5                 # robust z-score above this = eye closing
    blink_min_frames: int = 1
    blink_max_s: float = 0.5             # longer closures are counted as closures, not blinks
    blink_refractory_s: float = 0.33     # minimum gap between two events
    closure_max_s: float = 3.0           # longer runs are treated as signal loss, not closure
    min_fps: float = 15.0
    min_usable_share: float = 0.60
    min_duration_s: float = 10.0

    @classmethod
    def load(cls, path: str | Path | None) -> "Thresholds":
        if path is None:
            return cls()
        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown threshold keys: {sorted(unknown)}")
        return cls(**data)
