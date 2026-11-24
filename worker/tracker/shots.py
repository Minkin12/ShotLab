"""Shot segmentation, make/miss classification and ball-flight physics.

Works purely on per-frame detections (FrameData), so it can be tested with
synthetic data. Image coordinates: x to the right, y DOWN.
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right

import numpy as np

from .config import Config
from .frames import Box, FrameData, Shot

G = 9.81  # m/s^2
RIM_DIAMETER_M = 0.4572  # 18 in

TrackPoint = tuple[int, float, float, float]  # frame, cx, cy, ball width


# --------------------------------------------------------------------------- #
# Ball track & rim
# --------------------------------------------------------------------------- #
def clean_track(frames: list[FrameData], fps: float) -> list[TrackPoint]:
    """Ball centre per frame, dropping detections that jump implausibly far."""
    track: list[TrackPoint] = []
    reset_gap = max(1, int(fps * 0.5))
    for f in frames:
        b = f.ball
        if b is None:
            continue
        p = (f.idx, b.cx, b.cy, b.w)
        if track:
            last = track[-1]
            gap = p[0] - last[0]
            if gap <= reset_gap:
                max_jump = 4.0 * max(b.w, last[3], 1.0) * gap
                if math.hypot(p[1] - last[1], p[2] - last[2]) > max_jump:
                    continue  # outlier (another ball, a head, a light...)
        track.append(p)
    return track


class RimModel:
    """Median rim box over a sliding window - the rim barely moves, detections jitter."""

    def __init__(self, frames: list[FrameData], fps: float, window_s: float = 2.0):
        dets = [(f.idx, f.rim) for f in frames if f.rim is not None]
        self.idx = [d[0] for d in dets]
        self.boxes = np.array([[r.x1, r.y1, r.x2, r.y2] for _, r in dets], dtype=float).reshape(-1, 4)
        self.window = max(1, int(window_s * fps))
        self.fallback = max(1, int(10 * fps))

    def at(self, frame: int) -> Box | None:
        if not self.idx:
            return None
        for win in (self.window, self.fallback):
            lo = bisect_left(self.idx, frame - win)
            hi = bisect_right(self.idx, frame + win)
            if hi > lo:
                m = np.median(self.boxes[lo:hi], axis=0)
                return Box(*m.tolist())
        return None


# --------------------------------------------------------------------------- #
# Shot segmentation
# --------------------------------------------------------------------------- #
def detect_shots(frames: list[FrameData], fps: float, cfg: Config) -> list[Shot]:
    """Find attempts: the ball rises above the rim near it, then drops below it.

    Make/miss: where the ball's centre crosses the rim plane on the way down
    (interpolated between frames). The *last* downward crossing counts, so
    rim bounces are judged by where the ball finally drops.
    """
    track = clean_track(frames, fps)
    rims = RimModel(frames, fps)
    shots: list[Shot] = []

    max_flight = int(cfg.max_flight_s * fps)
    cooldown = int(cfg.shot_cooldown_s * fps)
    state, up_i, blocked_until = "idle", -1, -1

    for i, (fi, cx, cy, bw) in enumerate(track):
        if fi <= blocked_until:
            continue
        rim = rims.at(fi)
        if rim is None:
            continue
        above = cy < rim.y1 and abs(cx - rim.cx) < 4.0 * rim.w
        below = cy > rim.y2 + 0.5 * bw

        if state == "idle":
            if above:
                state, up_i = "up", i
            continue

        # state == "up"
        if fi - track[up_i][0] > max_flight:
            state = "idle"  # lost it / ball held overhead - not a shot
            continue
        if not below:
            continue

        shot = _classify(track, up_i, i, rim, cfg, len(shots) + 1, fps)
        if shot is not None:
            shots.append(shot)
        state, blocked_until = "idle", fi + cooldown

    return shots


def _classify(
    track: list[TrackPoint], up_i: int, end_i: int, rim: Box, cfg: Config, number: int, fps: float
) -> Shot | None:
    crossing = None
    for a, b in zip(track[up_i:end_i], track[up_i + 1 : end_i + 1]):
        if a[2] <= rim.cy < b[2]:  # moving down through the rim plane
            crossing = (a, b)
    if crossing is None:
        return None
    a, b = crossing
    frac = (rim.cy - a[2]) / (b[2] - a[2])
    x_cross = a[1] + frac * (b[1] - a[1])
    offset = abs(x_cross - rim.cx) / (rim.w / 2)

    up_frame = track[up_i][0]
    lo = up_frame - int(2 * fps)
    window = [p for p in track[: end_i + 1] if p[0] >= lo]
    return Shot(
        number=number,
        up_frame=up_frame,
        cross_frame=b[0],
        end_frame=track[end_i][0],
        made=offset <= cfg.make_tolerance,
        rim=rim,
        cross_offset=round(offset, 3),
        track=window,
    )


# --------------------------------------------------------------------------- #
# Ball flight
# --------------------------------------------------------------------------- #
def _residual(pts: list[TrackPoint], fps: float, probe: TrackPoint) -> float:
    f = np.array([p[0] for p in pts], dtype=float) / fps
    cx = np.polyfit(f, [p[1] for p in pts], 1)
    cy = np.polyfit(f, [p[2] for p in pts], 2)
    t = probe[0] / fps
    return math.hypot(np.polyval(cx, t) - probe[1], np.polyval(cy, t) - probe[2])


def flight_start(shot: Shot, fps: float) -> int:
    """Fallback release estimate when there's no pose.

    Fit the clearly-airborne part (above the rim -> rim crossing), then extend
    backwards one detection at a time while each earlier point still sits on
    the same parabola. The first point that doesn't is the ball in the hands.
    """
    flight = [p for p in shot.track if shot.up_frame <= p[0] <= shot.cross_frame]
    earlier = [p for p in shot.track if p[0] < shot.up_frame]
    if len(flight) < 4:
        return earlier[-1][0] if earlier else shot.up_frame
    for p in reversed(earlier):
        if _residual(flight, fps, p) > 0.35 * p[3]:
            break
        flight.insert(0, p)
    return flight[0][0]


def fit_flight(shot: Shot, fps: float) -> None:
    """Fit x(t) linear, y(t) quadratic from release to the rim crossing.

    Fills launch/entry angle, release speed and apex height. Angles are true
    angles only if the camera looks at the shot side-on (perpendicular to the
    shooter->rim line); at other angles they are consistent but inflated.
    Metres per pixel comes from gravity (fitted vertical acceleration),
    falling back to the rim width if the fit looks off.
    """
    start = shot.start_frame if shot.start_frame is not None else flight_start(shot, fps)
    shot.start_frame = start
    pts = [p for p in shot.track if start <= p[0] <= shot.cross_frame]
    if len(pts) < 6:
        return

    f = np.array([p[0] for p in pts], dtype=float)
    t = (f - f[0]) / fps
    x = np.array([p[1] for p in pts])
    y = np.array([p[2] for p in pts])

    vx, _ = np.polyfit(t, x, 1)
    a2, a1, a0 = np.polyfit(t, y, 2)
    ay = 2 * a2  # px/s^2, positive = downward
    if ay <= 0:
        return  # not a ballistic arc

    rim_scale = RIM_DIAMETER_M / max(shot.rim.w, 1.0)
    grav_scale = G / ay
    scale = grav_scale if 0.5 * rim_scale <= grav_scale <= 2.0 * rim_scale else rim_scale

    vy0 = a1  # dy/dt at t=0 (negative = upward)
    t_cross = t[-1]
    vy_cross = 2 * a2 * t_cross + a1
    t_apex = -a1 / (2 * a2)
    y_apex = a2 * t_apex**2 + a1 * t_apex + a0
    hvx = abs(vx) or 1e-6

    shot.launch_angle_deg = round(math.degrees(math.atan2(-vy0, hvx)), 1)
    shot.entry_angle_deg = round(math.degrees(math.atan2(vy_cross, hvx)), 1)
    shot.release_speed_mps = round(math.hypot(vx, vy0) * scale, 2)
    shot.apex_above_rim_m = round((shot.rim.cy - y_apex) * scale, 2)
