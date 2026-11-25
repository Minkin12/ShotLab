"""Shot segmentation, make/miss classification and ball-flight physics.

Works purely on per-frame detections (FrameData), so it can be tested with
synthetic data. Image coordinates: x to the right, y DOWN.
"""
from __future__ import annotations

import logging
import math
from bisect import bisect_left, bisect_right

import numpy as np

from .config import Config
from .frames import Box, FrameData, Shot

log = logging.getLogger(__name__)

G = 9.81  # m/s^2
RIM_DIAMETER_M = 0.4572  # 18 in

# Ball-flight quality gate: only report arc metrics from a well-sampled, ballistic track
MIN_FLIGHT_POINTS = 8  # ball detections between release and the rim
MIN_COVERAGE = 0.6  # detections must span >= 60% of the flight time
MAX_GAP = 0.35  # no hole bigger than 35% of the flight time
MIN_EACH_SIDE_OF_APEX = 2  # points on the way up AND on the way down
MAX_RMS_BALL_WIDTHS = 0.5  # fit residual, in ball widths
MAX_TRIM = 0.2  # may drop up to 20% of flight points as outliers...
TRIM_ABOVE_BALL_WIDTHS = 1.0  # ...if they sit more than 1 ball width off the arc

TrackPoint = tuple[int, float, float, float]  # frame, cx, cy, ball width


def rim_plane(rim: Box) -> float:
    """y of the rim opening. Many datasets box the rim *and net* together, so the
    box centre sits mid-net; the opening is near the top of such a tall box."""
    return rim.y1 + min(rim.h / 2, 0.15 * rim.w)


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
    plane = rim_plane(rim)
    crossings = [
        (a, b)
        for a, b in zip(track[up_i:end_i], track[up_i + 1 : end_i + 1])
        if a[2] <= plane < b[2]  # moving down through the rim plane
    ]
    if not crossings:
        return None
    a, b = crossings[-1]  # last crossing decides make/miss (after any rim bounces)
    frac = (plane - a[2]) / (b[2] - a[2])
    x_cross = a[1] + frac * (b[1] - a[1])
    offset = abs(x_cross - rim.cx) / (rim.w / 2)

    up_frame = track[up_i][0]
    lo = up_frame - int(2 * fps)
    window = [p for p in track[: end_i + 1] if p[0] >= lo]
    return Shot(
        number=number,
        up_frame=up_frame,
        cross_frame=b[0],
        # end of free flight: first time the ball gets within half a ball width of
        # the rim height (where it can first touch the rim)
        first_cross_frame=next(
            (p[0] for p in track[up_i : end_i + 1] if p[2] >= plane - 0.5 * p[3]), b[0]
        ),
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
    end = shot.first_cross_frame or shot.cross_frame
    flight = [p for p in shot.track if shot.up_frame <= p[0] <= end]
    earlier = [p for p in shot.track if p[0] < shot.up_frame]
    if len(flight) < 4:
        return earlier[-1][0] if earlier else shot.up_frame
    for p in reversed(earlier):
        if _residual(flight, fps, p) > 0.35 * p[3]:
            break
        flight.insert(0, p)
    return flight[0][0]


def _fit(pts: list[TrackPoint], fps: float, start: int):
    """x(t) linear, y(t) quadratic; t in seconds since release. Returns coeffs, t, residuals."""
    t = (np.array([p[0] for p in pts], dtype=float) - start) / fps
    x = np.array([p[1] for p in pts])
    y = np.array([p[2] for p in pts])
    cx = np.polyfit(t, x, 1)
    cy = np.polyfit(t, y, 2)
    resid = np.hypot(x - np.polyval(cx, t), y - np.polyval(cy, t))
    return cx, cy, t, resid


def fit_flight(shot: Shot, fps: float) -> None:
    """Fit the free flight: release -> first time the ball reaches the rim.

    Rim bounces and rattles after that point aren't ballistic, so they are left
    out. A few stray detections (ball half-hidden by a hand or the net) are
    trimmed. Arc metrics are only filled in when what remains passes the
    quality gate; otherwise they stay None and the reason is logged.

    Angles are true angles only if the camera looks at the shot side-on
    (perpendicular to the shooter->rim line); at other angles they are
    consistent but inflated. Metres per pixel comes from gravity (fitted
    vertical acceleration), checked against the rim width.
    """
    start = shot.start_frame if shot.start_frame is not None else flight_start(shot, fps)
    shot.start_frame = start
    end = shot.first_cross_frame or shot.cross_frame
    pts = [p for p in shot.track if start <= p[0] <= end]

    def skip(reason: str) -> None:
        log.info("shot %d: arc metrics skipped (%s)", shot.number, reason)

    if len(pts) < MIN_FLIGHT_POINTS:
        return skip(f"{len(pts)} ball detections in flight, need {MIN_FLIGHT_POINTS}")

    # drop the worst outliers one at a time
    ball_w = float(np.median([p[3] for p in pts]))
    for _ in range(int(MAX_TRIM * len(pts))):
        *_, resid = _fit(pts, fps, start)
        worst = int(np.argmax(resid))
        if resid[worst] <= TRIM_ABOVE_BALL_WIDTHS * ball_w or len(pts) <= MIN_FLIGHT_POINTS:
            break
        pts.pop(worst)

    flight = max(end - start, 1)
    frames_ = [p[0] for p in pts]
    coverage = (frames_[-1] - frames_[0]) / flight
    if coverage < MIN_COVERAGE:
        return skip(f"detections cover {coverage:.0%} of the flight")
    gap = max(b - a for a, b in zip(frames_, frames_[1:])) / flight
    if gap > MAX_GAP:
        return skip(f"ball lost for {gap:.0%} of the flight")

    (vx, _), (a2, a1, a0), t, resid = _fit(pts, fps, start)
    ay = 2 * a2  # px/s^2, positive = downward
    if ay <= 0:
        return skip("track is not a ballistic arc")

    t_apex = -a1 / (2 * a2)
    up, down = int((t < t_apex).sum()), int((t > t_apex).sum())
    if min(up, down) < MIN_EACH_SIDE_OF_APEX:
        return skip(f"{up} points before the apex, {down} after")

    rms = float(np.sqrt(np.mean(resid**2))) / max(ball_w, 1.0)
    if rms > MAX_RMS_BALL_WIDTHS:
        return skip(f"poor fit ({rms:.2f} ball widths off)")

    rim_scale = RIM_DIAMETER_M / max(shot.rim.w, 1.0)
    grav_scale = G / ay
    if not 0.5 * rim_scale <= grav_scale <= 2.0 * rim_scale:
        return skip("arc's gravity doesn't match the rim size")
    scale = grav_scale

    vy0 = a1  # dy/dt at release (negative = upward)
    vy_cross = 2 * a2 * t[-1] + a1
    y_apex = a2 * t_apex**2 + a1 * t_apex + a0
    hvx = abs(vx) or 1e-6

    shot.launch_angle_deg = round(math.degrees(math.atan2(-vy0, hvx)), 1)
    shot.entry_angle_deg = round(math.degrees(math.atan2(vy_cross, hvx)), 1)
    shot.release_speed_mps = round(math.hypot(vx, vy0) * scale, 2)
    shot.apex_above_rim_m = round((rim_plane(shot.rim) - y_apex) * scale, 2)