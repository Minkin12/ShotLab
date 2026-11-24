"""Glue: frames -> shots with flight + form metrics."""
from __future__ import annotations

from .config import Config
from .form import analyze_shot
from .frames import FrameData, Shot
from .shots import detect_shots, fit_flight


def analyze_frames(frames: list[FrameData], fps: float, cfg: Config) -> list[Shot]:
    by_idx = {f.idx: f for f in frames}
    shots = detect_shots(frames, fps, cfg)
    for s in shots:
        analyze_shot(by_idx, s, fps, cfg.shooting_hand)  # release + form
        fit_flight(s, fps)  # uses release as start when available
    return shots


def shot_row(s: Shot, fps: float) -> dict:
    """Flat dict matching the shots table columns."""
    ref = s.release_frame if s.release_frame is not None else (s.start_frame or s.up_frame)
    row = {
        "shot_number": s.number,
        "video_offset_s": round(ref / fps, 2),
        "made": s.made,
        "rim_offset": s.cross_offset,
        "launch_angle_deg": s.launch_angle_deg,
        "entry_angle_deg": s.entry_angle_deg,
        "release_speed_mps": s.release_speed_mps,
        "apex_above_rim_m": s.apex_above_rim_m,
        "shooting_hand": None,
        "release_elbow_angle": None,
        "set_point_elbow_angle": None,
        "min_knee_angle": None,
        "dip_to_release_ms": None,
        "release_height_ratio": None,
        "torso_lean_deg": None,
        "jump_ratio": None,
        "form_confidence": None,
        "clip_path": None,
    }
    row.update({k: v for k, v in s.form.items() if k in row})
    return row
