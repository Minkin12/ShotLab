"""Per-frame detection: ball + rim (custom YOLO model) and shooter pose (YOLO pose)."""
from __future__ import annotations

import logging
import math
from collections.abc import Callable

import cv2
import numpy as np

from .config import Config
from .frames import Box, FrameData, Pose, VideoInfo

log = logging.getLogger(__name__)


def video_info(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    info = VideoInfo(
        path=path,
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


class Detector:
    def __init__(self, cfg: Config):
        from ultralytics import YOLO  # heavy import, keep it local

        self.cfg = cfg
        self.obj = YOLO(cfg.object_model)
        self.pose = YOLO(cfg.pose_model)
        names = {int(k): v.lower() for k, v in self.obj.names.items()}
        self.ball_ids = {i for i, n in names.items() if n in cfg.ball_classes}
        self.rim_ids = {i for i, n in names.items() if n in cfg.rim_classes}
        if not self.ball_ids or not self.rim_ids:
            raise RuntimeError(
                f"object model classes {names} don't match BALL_CLASSES={cfg.ball_classes} "
                f"/ RIM_CLASSES={cfg.rim_classes}"
            )
        log.info("ball classes %s, rim classes %s", self.ball_ids, self.rim_ids)
        self.half = cfg.half and str(cfg.device).lower() != "cpu"
        log.info("inference: device=%s imgsz=%d fp16=%s", cfg.device, cfg.imgsz, self.half)

    def run(
        self, path: str, progress: Callable[[int, int], None] | None = None
    ) -> tuple[VideoInfo, list[FrameData]]:
        info = video_info(path)
        cap = cv2.VideoCapture(path)
        frames: list[FrameData] = []
        idx = 0
        last_ball: Box | None = None
        while True:
            ok, img = cap.read()
            if not ok:
                break
            fd = FrameData(idx=idx, t=idx / info.fps)
            fd.ball, fd.rim = self._objects(img, last_ball)
            fd.pose = self._shooter(img, fd.ball or last_ball)
            if fd.ball is not None:
                last_ball = fd.ball
            frames.append(fd)
            idx += 1
            if progress and idx % 300 == 0:
                progress(idx, info.frame_count)
        cap.release()
        info.frame_count = idx
        return info, frames

    def _objects(self, img: np.ndarray, last_ball: Box | None) -> tuple[Box | None, Box | None]:
        cfg = self.cfg
        r = self.obj.predict(
            img, conf=min(cfg.ball_conf, cfg.rim_conf), imgsz=cfg.imgsz, device=cfg.device,
            half=self.half, verbose=False,
        )[0]
        balls, rims = [], []
        for (x1, y1, x2, y2), c, k in zip(
            r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()
        ):
            b = Box(x1, y1, x2, y2, c)
            if int(k) in self.ball_ids and c >= cfg.ball_conf:
                balls.append(b)
            elif int(k) in self.rim_ids and c >= cfg.rim_conf:
                rims.append(b)
        rim = max(rims, key=lambda b: b.conf) if rims else None
        if not balls:
            return None, rim
        if last_ball is None or len(balls) == 1:
            return max(balls, key=lambda b: b.conf), rim
        # several balls on court: follow the one nearest the last position
        return min(balls, key=lambda b: math.hypot(b.cx - last_ball.cx, b.cy - last_ball.cy)), rim

    def _shooter(self, img: np.ndarray, ball: Box | None) -> Pose | None:
        cfg = self.cfg
        r = self.pose.predict(
            img, conf=cfg.pose_conf, imgsz=cfg.imgsz, device=cfg.device, half=self.half, verbose=False
        )[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return None
        kpts = r.keypoints.data.cpu().numpy()  # (n, 17, 3)
        boxes = [Box(*b, c) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
        if ball is None:
            i = max(range(len(boxes)), key=lambda j: boxes[j].w * boxes[j].h)
        else:
            # the person whose wrist is closest to the ball
            def wrist_dist(j: int) -> float:
                ds = [
                    math.hypot(kpts[j][w][0] - ball.cx, kpts[j][w][1] - ball.cy)
                    for w in (9, 10)
                    if kpts[j][w][2] > 0.3
                ]
                return min(ds) if ds else math.inf

            i = min(range(len(boxes)), key=wrist_dist)
            if wrist_dist(i) == math.inf:
                i = max(range(len(boxes)), key=lambda j: boxes[j].w * boxes[j].h)
        return Pose(kpts=kpts[i], box=boxes[i])