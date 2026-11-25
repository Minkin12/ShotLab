"""Glue: frames -> shots with flight + form metrics."""
from __future__ import annotations

from .config import Config
from .form import analyze_shot
from .frames import FrameData, Shot
from .shots import detect_shots, fit_flight, rim_plane


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


def dump_tracks(frames: list[FrameData], fps: float, shots: list[Shot]) -> dict:
    """Per-shot raw ball detections and rim geometry, for offline analysis.

    Each shot covers 2 s before the ball goes above the rim to 1 s after it is
    clearly below it, so the drop through (or past) the net is included.
    """
    by_idx = {f.idx: f for f in frames}
    out = {"fps": fps, "shots": []}
    for s in shots:
        lo, hi = s.up_frame - int(2 * fps), s.end_frame + int(fps)
        balls = []
        for i in range(lo, hi + 1):
            f = by_idx.get(i)
            if f is not None and f.ball is not None:
                b = f.ball
                balls.append([i, round(b.cx, 1), round(b.cy, 1), round(b.w, 1), round(b.h, 1), round(b.conf, 3)])
        r = s.rim
        out["shots"].append({
            "shot_number": s.number,
            "made": bool(s.made),
            "rim_offset": s.cross_offset,
            "rim": [round(v, 1) for v in (r.x1, r.y1, r.x2, r.y2)],
            "rim_plane_y": round(rim_plane(r), 1),
            "frames": {
                "release": s.release_frame,
                "start": s.start_frame,
                "up": s.up_frame,
                "first_cross": s.first_cross_frame,
                "cross": s.cross_frame,
                "end": s.end_frame,
            },
            "ball": balls,  # [frame, cx, cy, w, h, conf]
        })
    return out