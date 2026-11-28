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
# physically plausible jump-shot arcs; anything outside is a bad fit, not a real shot
LAUNCH_ANGLE_RANGE = (25.0, 75.0)  # degrees
ENTRY_ANGLE_RANGE = (20.0, 80.0)  # degrees
RELEASE_SPEED_RANGE = (2.5, 15.0)  # m/s
EXIT_WINDOW_S = 0.2  # exit speed = fall speed over this long after the final rim crossing
# Session gravity check: within a session, the metres-per-pixel that gravity implies is
# similar for every shot, varying only with the shot's distance from the camera. A shot whose
# own value is far from the session's median is a bad fit, not a real arc. (The rim and ball
# boxes are no use as rulers: the model draws both ~1.5x bigger than the real objects.)
MIN_CALIBRATION_SHOTS = 5
GRAVITY_TOLERANCE = 0.6  # own scale within 1/1.6x .. 1.6x of the median (distance varies)

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
            shot.exit_speed_mps = exit_speed(track, shot.cross_frame, rim, fps)
            limit = cfg.max_make_exit_speed
            if shot.made and limit is not None and shot.exit_speed_mps is not None and shot.exit_speed_mps > limit:
                shot.made = False
                log.info(
                    "shot %d: make overruled - ball left the rim at %.2f m/s (MAX_MAKE_EXIT_SPEED %.2f)",
                    shot.number, shot.exit_speed_mps, limit,
                )
        if shot is not None:
            shots.append(shot)
        state, blocked_until = "idle", fi + cooldown

    return shots


CONTACT_TOLERANCE = 0.5  # ball widths off the shot's own arc = it hit the rim or backboard
APEX_WINDOW = 5  # detections each side that must be lower for a point to count as the apex


def _first_apex(seg: list[TrackPoint]) -> int:
    """Index of the shot's own apex: the first local top of the arc.

    Not simply the highest point - a ball that bounces high off the rim can go
    higher than the shot did.
    """
    ys = [p[2] for p in seg]
    for i in range(len(ys)):
        lo, hi = max(0, i - APEX_WINDOW), i + APEX_WINDOW + 1
        if ys[i] <= min(ys[lo:hi]):
            return i
    return min(range(len(ys)), key=ys.__getitem__)


def _contact_frame(seg: list[TrackPoint], plane: float, fps: float) -> int | None:
    """Last frame on the shot's own arc before it hits something, or None if it never leaves it.

    Fits the clearly-free part of the flight (up to two ball widths above the rim
    on the way down) and follows the ball forward while it stays on that parabola.
    The ball often meets the front or top of the rim with its centre still a little
    above the rim plane, which the height test in _classify can miss; without this
    the bounce afterwards ends up in the arc fit.
    """
    apex = _first_apex(seg)
    k = next((i for i in range(apex, len(seg)) if seg[i][2] >= plane - 2 * seg[i][3]), None)
    if k is None or k < 6:
        return None
    base, rest = list(seg[:k]), seg[k:]
    for i, p in enumerate(rest):
        if _residual(base, fps, p) > CONTACT_TOLERANCE * p[3]:
            nxt = rest[i + 1] if i + 1 < len(rest) else None
            if nxt is None or _residual(base, fps, nxt) > CONTACT_TOLERANCE * nxt[3]:
                return base[-1][0]
            continue  # one stray detection: skip it
        base.append(p)
    return None


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
    # end of free flight: first time, AFTER the top of the arc, that the ball gets
    # within half a ball width of the rim height (where it can first touch the rim).
    # Searching only after the apex matters: on the way up the ball passes the same
    # height, and with a tall rim(+net) box that is right where the shot starts.
    # The apex is the shot's own first top (a high bounce can go higher), and the
    # flight also ends earlier if the ball leaves its arc before getting that low.
    seg = track[up_i : end_i + 1]
    apex = _first_apex(seg)
    first_reach = next((p[0] for p in seg[apex:] if p[2] >= plane - 0.5 * p[3]), b[0])
    contact = _contact_frame(seg, plane, fps)
    if contact is not None:
        first_reach = min(first_reach, contact)
    return Shot(
        number=number,
        up_frame=up_frame,
        cross_frame=b[0],
        first_cross_frame=first_reach,
        end_frame=track[end_i][0],
        made=offset <= cfg.make_tolerance,
        rim=rim,
        cross_offset=round(offset, 3),
        track=window,
    )


def exit_speed(track: list[TrackPoint], cross_frame: int, rim: Box, fps: float) -> float | None:
    """Downward speed (m/s) of the ball in the EXIT_WINDOW_S after its final rim crossing.

    Through the net the ball is slowed (about 1 m/s on the first test session);
    a ball rattling out and dropping just in front of / behind the rim falls
    freely (about 2+ m/s). Scale comes from the rim width.
    """
    hi = cross_frame + round(EXIT_WINDOW_S * fps)
    pts = [p for p in track if cross_frame <= p[0] <= hi]
    if len(pts) < 3:
        return None
    t = np.array([p[0] for p in pts], dtype=float) / fps
    vy = np.polyfit(t, [p[2] for p in pts], 1)[0]  # px/s, positive = falling
    return round(float(vy) * RIM_DIAMETER_M / max(rim.w, 1.0), 2)


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
    trimmed. Launch/entry angles are only filled in when what remains passes the
    quality gate; otherwise they stay None and the reason is logged. Speed and
    apex height need the session's pixel scale and are filled in afterwards by
    calibrate_flights (from the raw values left in shot.flight_px).

    Angles are true angles only if the camera looks at the shot side-on
    (perpendicular to the shooter->rim line); at other angles they are
    consistent but inflated.
    """
    start = shot.start_frame if shot.start_frame is not None else flight_start(shot, fps)
    shot.start_frame = start
    end = shot.first_cross_frame or shot.cross_frame
    pts = [p for p in shot.track if start <= p[0] <= end]

    def skip(reason: str) -> None:
        log.info("shot %d: arc metrics skipped (%s)", shot.number, reason)

    if len(pts) < MIN_FLIGHT_POINTS:
        return skip(f"{len(pts)} ball detections in flight, need {MIN_FLIGHT_POINTS}")

    # drop the worst outliers one at a time - but never thin out the side of the
    # apex (rising or falling) that is already down to its minimum number of points
    ball_w = float(np.median([p[3] for p in pts]))
    for _ in range(int(MAX_TRIM * len(pts))):
        _, (a2, a1, _), t, resid = _fit(pts, fps, start)
        worst = int(np.argmax(resid))
        if resid[worst] <= TRIM_ABOVE_BALL_WIDTHS * ball_w or len(pts) <= MIN_FLIGHT_POINTS:
            break
        if a2 > 0:
            t_apex = -a1 / (2 * a2)
            side = t < t_apex if t[worst] < t_apex else t > t_apex
            if int(side.sum()) <= MIN_EACH_SIDE_OF_APEX:
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

    vy0 = a1  # dy/dt at release (negative = upward)
    vy_cross = 2 * a2 * t[-1] + a1
    y_apex = a2 * t_apex**2 + a1 * t_apex + a0
    hvx = abs(vx) or 1e-6

    # angles don't depend on the pixel scale; speed and apex height do (see calibrate_flights)
    launch = math.degrees(math.atan2(-vy0, hvx))
    entry = math.degrees(math.atan2(vy_cross, hvx))
    if not (
        LAUNCH_ANGLE_RANGE[0] <= launch <= LAUNCH_ANGLE_RANGE[1]
        and ENTRY_ANGLE_RANGE[0] <= entry <= ENTRY_ANGLE_RANGE[1]
    ):
        return skip(f"implausible arc (launch {launch:.0f} deg, entry {entry:.0f} deg)")

    shot.launch_angle_deg = round(launch, 1)
    shot.entry_angle_deg = round(entry, 1)
    shot.flight_px = {"ay": float(ay), "v0": float(math.hypot(vx, vy0)), "apex": float(rim_plane(shot.rim) - y_apex)}


def _clear_flight(shot: Shot, reason: str) -> None:
    log.info("shot %d: arc metrics skipped (%s)", shot.number, reason)
    shot.launch_angle_deg = shot.entry_angle_deg = None
    shot.release_speed_mps = shot.apex_above_rim_m = None
    shot.flight_px = None


def calibrate_flights(shots: list[Shot]) -> float | None:
    """Turn each fitted arc's pixel values into metres and sanity-check its gravity.

    Each shot uses its own gravity-based scale (G / fitted vertical acceleration):
    that is measured where the ball actually flew, so it is right even when shots
    are taken at different distances from the camera. The check compares it with
    the session median (with at least MIN_CALIBRATION_SHOTS arcs), otherwise with
    the rim width as before. Returns the session median scale (m/px) or None.
    """
    fitted = [s for s in shots if s.flight_px]
    own = {s.number: G / s.flight_px["ay"] for s in fitted}
    session = float(np.median(list(own.values()))) if len(fitted) >= MIN_CALIBRATION_SHOTS else None
    if session is not None:
        log.info("session scale: 1 px = %.1f mm (gravity, median of %d arcs)", session * 1000, len(fitted))

    for s in fitted:
        scale = own[s.number]
        if session is not None:
            ratio = scale / session
            if not 1 / (1 + GRAVITY_TOLERANCE) <= ratio <= 1 + GRAVITY_TOLERANCE:
                _clear_flight(s, f"arc's gravity is {ratio:.2f}x the session's")
                continue
        else:
            rim_scale = RIM_DIAMETER_M / max(s.rim.w, 1.0)
            if not 0.5 * rim_scale <= scale <= 2.0 * rim_scale:
                _clear_flight(s, "arc's gravity doesn't match the rim size")
                continue

        speed = s.flight_px["v0"] * scale
        if not RELEASE_SPEED_RANGE[0] <= speed <= RELEASE_SPEED_RANGE[1]:
            _clear_flight(s, f"implausible release speed ({speed:.1f} m/s)")
            continue
        s.release_speed_mps = round(speed, 2)
        s.apex_above_rim_m = round(s.flight_px["apex"] * scale, 2)
    return session
