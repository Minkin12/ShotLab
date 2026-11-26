"""Plain data types shared by detection and analysis.

Kept free of heavy dependencies (only numpy) so the analysis code can be
unit-tested without a GPU, YOLO weights or video files.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1


@dataclass
class Pose:
    """One person's COCO-17 keypoints as an array of shape (17, 3): x, y, conf."""

    kpts: np.ndarray
    box: Box | None = None

    def point(self, i: int, min_conf: float = 0.3) -> np.ndarray | None:
        x, y, c = self.kpts[i]
        if c < min_conf:
            return None
        return np.array([x, y], dtype=float)

    def conf(self, i: int) -> float:
        return float(self.kpts[i][2])


@dataclass
class FrameData:
    idx: int
    t: float  # seconds from start of video
    ball: Box | None = None
    rim: Box | None = None
    pose: Pose | None = None  # the most likely shooter


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


@dataclass
class Shot:
    number: int
    up_frame: int  # first frame the ball is above the rim
    cross_frame: int  # frame where the ball crossed the rim plane going down (last time)
    end_frame: int  # first frame the ball is clearly below the rim
    made: bool
    rim: Box
    cross_offset: float  # |x at rim plane - rim centre| / (rim width / 2)
    # (frame, cx, cy, ball_w) from ~2 s before up_frame through end_frame
    track: list[tuple[int, float, float, float]] = field(default_factory=list)
    first_cross_frame: int | None = None  # first time it reached rim height (end of free flight)
    start_frame: int | None = None  # start of free flight (release if known)
    release_frame: int | None = None
    exit_speed_mps: float | None = None  # how fast the ball drops in the 0.2 s after the rim
    # ball-flight metrics
    launch_angle_deg: float | None = None
    entry_angle_deg: float | None = None
    release_speed_mps: float | None = None
    apex_above_rim_m: float | None = None
    # raw fit in pixels {"ay": px/s^2, "v0": px/s, "apex": px above rim}; set by
    # fit_flight, converted to metres by calibrate_flights once the session scale is known
    flight_px: dict | None = None
    # form metrics (see form.py), keyed by DB column name
    form: dict = field(default_factory=dict)
