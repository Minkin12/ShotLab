"""Make/miss classifier: a second opinion on shots the ball-position logic calls a make.

The model is a YOLO classification model trained on `tracker crops` images (classes
'make' and 'miss'). For every shot it gives make_prob; a "make" whose p(miss) is above
MAKE_CLASSIFIER_THRESHOLD is scored a miss. Shots already scored a miss are never
changed - the position logic is reliable for those.
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
        log.info("make classifier %s (threshold %.2f)", cfg.make_classifier, cfg.make_classifier_threshold)

    def p_miss(self, images: list[np.ndarray]) -> list[float]:
        if not images:
            return []
        out = []
        for i in range(0, len(images), 64):
            res = self.model.predict(images[i : i + 64], device=self.cfg.device, verbose=False)
            out += [float(r.probs.data[self.miss_id]) for r in res]
        return out


def apply_classifier(video: Path, shots: list[Shot], fps: float, threshold: float, clf) -> None:
    """Sets shot.make_prob for every shot and overrules makes the classifier is sure were misses.

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
        if s.made and pm > threshold:
            s.made = False
            log.info("shot %d: make overruled by classifier (p(miss) %.2f > %.2f)", s.number, pm, threshold)
