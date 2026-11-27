"""Make/miss classifier: a second opinion on shots the ball-position logic calls a make.

The model is a YOLO classification model trained on `tracker crops` images (classes
'make' and 'miss'). For every shot it gives make_prob; a "make" whose p(miss) is above
MAKE_CLASSIFIER_THRESHOLD is scored a miss.

Optional rescue (MAKE_CLASSIFIER_RESCUE): a near-miss - the ball came down through the
rim plane just outside MAKE_TOLERANCE, but within MAKE_CLASSIFIER_RESCUE_OFFSET - is
scored a make when make_prob is at least the rescue value. Clear misses and misses from
the exit-speed rule are never changed.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .crops import collect_rim_crops, stack_image
from .frames import Shot
from .shots import rim_plane

log = logging.getLogger(__name__)


def _as_training_jpeg(img: np.ndarray) -> np.ndarray:
    """Training images were saved as JPEG (quality 95); match that exactly."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


class MakeClassifier:
    def __init__(self, cfg: Config):
        from ultralytics import YOLO  # heavy import, keep it local

        self.cfg = cfg
        self.model = YOLO(cfg.make_classifier)
        names = {int(k): str(v).lower() for k, v in self.model.names.items()}
        if "miss" not in names.values() or "make" not in names.values():
            raise RuntimeError(f"make classifier must have classes 'make' and 'miss', got {names}")
        self.miss_id = next(k for k, v in names.items() if v == "miss")
        log.info(
            "make classifier %s (threshold %.2f, rescue %s)", cfg.make_classifier, cfg.make_classifier_threshold,
            f"{cfg.make_classifier_rescue:.2f} up to rim_offset {cfg.make_classifier_rescue_offset:.2f}"
            if cfg.make_classifier_rescue is not None else "off",
        )

    def p_miss(self, images: list[np.ndarray]) -> list[float]:
        if not images:
            return []
        out = []
        for i in range(0, len(images), 64):
            res = self.model.predict(images[i : i + 64], device=self.cfg.device, verbose=False)
            out += [float(r.probs.data[self.miss_id]) for r in res]
        return out


def verdict(
    made: bool,
    rim_offset: float | None,
    make_prob: float | None,
    threshold: float,
    make_tolerance: float,
    rescue: float | None = None,
    rescue_max_offset: float = 0.9,
) -> tuple[bool, str | None]:
    """The classifier's say on one shot. Returns (made, reason) - reason is None when unchanged.

    Shared by the tracker and `score`, so re-scored results match what processing would do.
    """
    if make_prob is None:
        return made, None
    if made and 1.0 - make_prob > threshold:
        return False, "overruled"
    if (
        not made
        and rescue is not None
        and rim_offset is not None
        and make_tolerance < rim_offset <= rescue_max_offset
        and make_prob >= rescue
    ):
        return True, "rescued"
    return made, None


def apply_classifier(
    video: Path,
    shots: list[Shot],
    fps: float,
    threshold: float,
    clf,
    make_tolerance: float = 0.6,
    rescue: float | None = None,
    rescue_max_offset: float = 0.9,
) -> None:
    """Sets shot.make_prob for every shot, overrules makes the classifier is sure were misses
    and, if `rescue` is set, rescues near-misses it is sure were makes.

    `clf` is anything with a p_miss(list of images) -> list of floats method.
    """
    if not shots:
        return
    specs = [(s.number, [s.rim.x1, s.rim.y1, s.rim.x2, s.rim.y2], rim_plane(s.rim), s.cross_frame) for s in shots]
    crops = collect_rim_crops(video, specs, fps)
    ready = [s for s in shots if all(c is not None and c.size for c in crops[s.number])]
    probs = clf.p_miss([_as_training_jpeg(stack_image(crops[s.number])) for s in ready])
    for s, pm in zip(ready, probs):
        s.make_prob = round(1.0 - pm, 3)
        s.made, why = verdict(s.made, s.cross_offset, s.make_prob, threshold, make_tolerance, rescue, rescue_max_offset)
        if why == "overruled":
            log.info("shot %d: make overruled by classifier (p(miss) %.2f > %.2f)", s.number, pm, threshold)
        elif why == "rescued":
            log.info("shot %d: miss rescued by classifier (make_prob %.2f >= %.2f, rim_offset %.2f)",
                     s.number, s.make_prob, rescue, s.cross_offset)
