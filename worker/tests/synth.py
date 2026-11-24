"""Synthetic shot generator: a right-handed shooter, a ballistic ball flight, a static rim."""
from __future__ import annotations

import math

import numpy as np

from tracker.frames import Box, FrameData, Pose

RIM = Box(970, 292.5, 1030, 307.5, 0.9)  # 60 px wide -> 0.00762 m/px
SCALE = 0.4572 / 60
G_PX = 9.81 / SCALE
BALL_W = 30.0
REL = np.array([400.0, 450.0])  # release point (~1.3 m below the rim)


def flight_speed(angle_deg: float, target: tuple[float, float]) -> float:
    th = math.radians(angle_deg)
    dx, dy = target[0] - REL[0], REL[1] - target[1]
    return math.sqrt(G_PX * dx**2 / (2 * math.cos(th) ** 2 * (dx * math.tan(th) - dy)))


def _pose(wrist: np.ndarray, knee_dx: float, elbow: np.ndarray | None = None) -> Pose:
    k = np.zeros((17, 3))
    pts = {
        0: (380, 450), 5: (370, 500), 6: (390, 500), 7: (360, 560), 9: (355, 610),
        11: (370, 620), 12: (390, 620), 13: (370, 700), 15: (370, 780), 16: (390, 780),
    }
    for i, (x, y) in pts.items():
        k[i] = (x, y, 0.9)
    k[14] = (390 + knee_dx, 700, 0.9)
    k[10] = (*wrist, 0.9)
    k[8] = (*(elbow if elbow is not None else (wrist + np.array([390, 500])) / 2 + [12, 0]), 0.9)
    return Pose(kpts=k)


def shot_frames(
    fps: float = 60.0,
    angle: float = 50.0,
    target_x: float = 1000.0,
    start: int = 0,
    release_at: float = 1.0,
    knee_dip_before: float = 0.4,
    with_pose: bool = True,
) -> tuple[list[FrameData], int]:
    """Frames for one shot. Returns (frames, true release frame index)."""
    v = flight_speed(angle, (target_x, RIM.cy))
    th = math.radians(angle)
    vx, vy = v * math.cos(th), -v * math.sin(th)
    rel_f = start + int(release_at * fps)
    frames = []
    total = int((release_at + 2.0) * fps)
    for n in range(total):
        i = start + n
        t = (i - rel_f) / fps
        # knee: triangular dip peaking knee_dip_before seconds before release (to 140 deg)
        tk = t + knee_dip_before
        dip = max(0.0, 1 - abs(tk) / 0.3)
        knee_dx = 80 * math.tan(math.radians(20)) * dip
        if t <= 0:
            # ball in hands, lifted from y=700 to release point over the last 0.3 s
            lift = min(1.0, max(0.0, (t + 0.3) / 0.3))
            ball = REL + np.array([0.0, 100.0 * (1 - lift)])
            wrist = ball + np.array([-5.0, 12.0])
        else:
            ball = REL + np.array([vx * t, vy * t + 0.5 * G_PX * t * t])
            wrist = REL + np.array([-5.0, 12.0])  # follow-through holds
        b = Box(ball[0] - BALL_W / 2, ball[1] - BALL_W / 2, ball[0] + BALL_W / 2, ball[1] + BALL_W / 2)
        if ball[1] > 900:  # off the bottom of the frame
            b = None
        pose = _pose(wrist, knee_dx) if with_pose else None
        frames.append(FrameData(idx=i, t=i / fps, ball=b, rim=RIM, pose=pose))
    return frames, rel_f
