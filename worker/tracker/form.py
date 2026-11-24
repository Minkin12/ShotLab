"""Shooting-form analysis from 2D pose keypoints (COCO-17).

Best with the camera side-on to the shooter. All metrics are 2D projections:
compare them with *your own* makes and misses rather than with textbook
numbers.

Metrics (DB column -> meaning):
  shooting_hand          'right' / 'left'
  release_elbow_angle    shoulder-elbow-wrist angle at release (180 = straight)
  set_point_elbow_angle  tightest elbow angle while the ball is above the
                         shoulder before release (the "set point")
  min_knee_angle         deepest knee bend in the 1 s before release
  dip_to_release_ms      time from deepest knee bend to release (rhythm)
  release_height_ratio   wrist height above ankle / nose height above ankle
                         at release (>1 = released above the head)
  torso_lean_deg         torso angle from vertical at release; + = toward
                         the rim, - = fading away
  jump_ratio             ankle lift at release / body height (0 = set shot)
  form_confidence        mean keypoint confidence of the joints used
"""
from __future__ import annotations

import math

import numpy as np

from .frames import FrameData, Shot

NOSE = 0
SIDES = {
    "left": dict(shoulder=5, elbow=7, wrist=9, hip=11, knee=13, ankle=15),
    "right": dict(shoulder=6, elbow=8, wrist=10, hip=12, knee=14, ankle=16),
}
HELD_DIST = 1.3  # wrist within 1.3 ball-widths of the ball centre = ball in hand


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n == 0:
        return float("nan")
    cos = float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _wrist_ball_dist(f: FrameData, side: str) -> float | None:
    if f.pose is None or f.ball is None:
        return None
    w = f.pose.point(SIDES[side]["wrist"])
    if w is None:
        return None
    return math.hypot(w[0] - f.ball.cx, w[1] - f.ball.cy) / max(f.ball.w, 1.0)


def _window(by_idx: dict[int, FrameData], lo: int, hi: int) -> list[FrameData]:
    return [by_idx[i] for i in range(lo, hi + 1) if i in by_idx]


def detect_hand(by_idx: dict[int, FrameData], shot: Shot, fps: float) -> str | None:
    """Which wrist stays closest to the ball in the 1 s before it goes up."""
    votes = {"left": 0, "right": 0}
    for f in _window(by_idx, shot.up_frame - int(fps), shot.up_frame):
        d = {s: _wrist_ball_dist(f, s) for s in votes}
        if d["left"] is not None and d["right"] is not None:
            votes["left" if d["left"] < d["right"] else "right"] += 1
    if votes["left"] == votes["right"]:
        return None
    return max(votes, key=votes.get)


def _elbow_angle(f: FrameData, hand: str) -> float | None:
    if f.pose is None:
        return None
    k = SIDES[hand]
    p = [f.pose.point(k["shoulder"]), f.pose.point(k["elbow"]), f.pose.point(k["wrist"])]
    return None if any(q is None for q in p) else joint_angle(*p)


def find_release(by_idx: dict[int, FrameData], shot: Shot, fps: float, hand: str) -> int | None:
    """Frame the ball leaves the shooting hand.  # ◀ CHANGED (docstring)

    1. Last frame the ball is detected in the hand before it rises above the rim.
    2. Ball detections often drop out during the fast final arm extension
       (motion blur), so step 1 can land on the set point. The real release
       happens at (near) full extension: if the elbow keeps opening by 15+ deg
       in the next 0.3 s, take the first frame where it reaches 97% of its peak.
    """
    held = None
    for f in _window(by_idx, shot.up_frame - int(1.5 * fps), shot.up_frame):
        d = _wrist_ball_dist(f, hand)
        if d is not None and d <= HELD_DIST:
            held = f.idx
    if held is None:
        return None

    angles = [
        (f.idx, a)
        for f in _window(by_idx, held, min(shot.up_frame, held + int(0.3 * fps)))
        if (a := _elbow_angle(f, hand)) is not None
    ]
    if not angles:
        return held
    angles = _smoothed(angles)
    peak = max(a for _, a in angles)
    if peak - angles[0][1] < 15:  # arm was already extended when the ball was last seen
        return held
    return next(fi for fi, a in angles if a >= 0.97 * peak)


def _smoothed(values: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Median of 3 over consecutive samples - pose angles are jittery."""
    out = []
    for i, (fi, _) in enumerate(values):
        nb = [v for _, v in values[max(0, i - 1) : i + 2]]
        out.append((fi, float(np.median(nb))))
    return out


def analyze_form(
    by_idx: dict[int, FrameData], shot: Shot, fps: float, hand: str, rim_cx: float
) -> dict:
    rel = shot.release_frame
    if rel is None or rel not in by_idx or by_idx[rel].pose is None:
        return {"shooting_hand": hand}
    k = SIDES[hand]
    load = _window(by_idx, rel - int(1.0 * fps), rel)
    out: dict = {"shooting_hand": hand}

    def pts(f: FrameData, *names: str):
        if f.pose is None:
            return None
        p = [f.pose.point(k[n]) if n in k else f.pose.point(NOSE) for n in names]
        return None if any(q is None for q in p) else p

    # elbow series and knee series over the load window
    elbow, knee, set_elbow = [], [], []
    for f in load:
        e = pts(f, "shoulder", "elbow", "wrist")
        if e:
            ang = joint_angle(*e)
            elbow.append((f.idx, ang))
            if e[2][1] < e[0][1]:  # wrist above shoulder
                set_elbow.append((f.idx, ang))
        kn = pts(f, "hip", "knee", "ankle")
        if kn:
            knee.append((f.idx, joint_angle(*kn)))

    elbow_s = dict(_smoothed(elbow))
    if rel in elbow_s:
        out["release_elbow_angle"] = round(elbow_s[rel], 1)
    if set_elbow:
        out["set_point_elbow_angle"] = round(min(v for _, v in _smoothed(set_elbow)), 1)
    if knee:
        ks = _smoothed(knee)
        kmin_f, kmin = min(ks, key=lambda p: p[1])
        out["min_knee_angle"] = round(kmin, 1)
        out["dip_to_release_ms"] = round((rel - kmin_f) / fps * 1000)

    # posture at release
    f = by_idx[rel]
    body = pts(f, "nose", "wrist", "ankle")
    if body:
        nose, wrist, ankle = body
        height = ankle[1] - nose[1]
        if height > 0:
            out["release_height_ratio"] = round((ankle[1] - wrist[1]) / height, 2)
            # jump: compare ankle with its lowest (standing) position in the load window
            ankles = [p[0][1] for g in load if (p := pts(g, "ankle"))]
            if ankles:
                out["jump_ratio"] = round(max(0.0, (max(ankles) - ankle[1]) / height), 3)

    ps = f.pose
    sh = [ps.point(5), ps.point(6)]
    hp = [ps.point(11), ps.point(12)]
    sh = [p for p in sh if p is not None]
    hp = [p for p in hp if p is not None]
    if sh and hp:
        s_mid, h_mid = np.mean(sh, axis=0), np.mean(hp, axis=0)
        dx, dy = s_mid[0] - h_mid[0], h_mid[1] - s_mid[1]  # dy > 0: shoulders above hips
        lean = math.degrees(math.atan2(abs(dx), dy))
        toward = (dx > 0) == (rim_cx > h_mid[0])
        out["torso_lean_deg"] = round(lean if toward else -lean, 1)

    used = [k["shoulder"], k["elbow"], k["wrist"], k["hip"], k["knee"], k["ankle"], NOSE]
    out["form_confidence"] = round(float(np.mean([ps.conf(i) for i in used])), 2)
    return out


def analyze_shot(by_idx: dict[int, FrameData], shot: Shot, fps: float, hand_cfg: str) -> None:
    """Sets shot.release_frame / shot.start_frame / shot.form in place."""
    hand = hand_cfg if hand_cfg in SIDES else detect_hand(by_idx, shot, fps)
    if hand is None:
        return
    shot.release_frame = find_release(by_idx, shot, fps, hand)
    if shot.release_frame is not None:
        shot.start_frame = shot.release_frame
    shot.form = analyze_form(by_idx, shot, fps, hand, shot.rim.cx)