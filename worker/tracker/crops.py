"""Make/miss classifier training data: one image of the rim per shot, labeled from your truth file.

  python -m tracker.main crops VIDEO TRACKS.json TRUTH.txt --out /models/datasets/makecls --split train

VIDEO        the session video
TRACKS.json  from `process ... --dump-tracks` (gives each shot's rim box and crossing frame)
TRUTH.txt    your make/miss list, same formats as `score`

Each image is a square crop around the rim built from three moments around the ball's
final rim crossing, stacked as the three colour channels (grayscale each):
    red   = 0.1 s before the crossing
    green = 0.1 s after
    blue  = 0.3 s after
Still things (rim, backboard, trees) come out gray; anything that moves - the ball, the
net snapping on a make - shows up as colour. One square image keeps it robust to the
crop/zoom augmentation that image classifiers train with.

Output follows the YOLO classification layout:  OUT/<split>/make/*.jpg, OUT/<split>/miss/*.jpg
With --preview, a side-by-side colour strip of the three moments is also written to
OUT/preview/<split>/ so you can eyeball the labels.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from .score import parse_truth, score

log = logging.getLogger(__name__)

STACK_TIMES_S = (-0.1, 0.1, 0.3)  # relative to the final rim crossing
CROP_RIM_WIDTHS = 2.5  # crop side, in rim-box widths
CROP_ABOVE = 0.4  # fraction of the crop above the rim plane (the rest is net and below)
IMG_SIZE = 224


def crop_box(rim: list[float], plane_y: float, width: int, height: int) -> tuple[int, int, int, int]:
    """Square around the rim: centred horizontally, mostly below the rim plane (the net)."""
    x1, _, x2, _ = rim
    side = CROP_RIM_WIDTHS * (x2 - x1)
    cx = (x1 + x2) / 2
    left, top = cx - side / 2, plane_y - CROP_ABOVE * side
    left = int(round(min(max(left, 0), max(width - side, 0))))
    top = int(round(min(max(top, 0), max(height - side, 0))))
    s = int(round(min(side, width, height)))
    return left, top, left + s, top + s


def stack_image(crops: list[np.ndarray]) -> np.ndarray:
    """Three BGR crops (oldest first) -> one image with each moment in its own channel."""
    gray = [cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), (IMG_SIZE, IMG_SIZE)) for c in crops]
    return cv2.merge([gray[2], gray[1], gray[0]])  # B, G, R  ->  red = earliest


def preview_strip(crops: list[np.ndarray], label: str) -> np.ndarray:
    tiles = [cv2.resize(c, (IMG_SIZE, IMG_SIZE)) for c in crops]
    strip = np.hstack(tiles)
    colour = (0, 200, 0) if label == "make" else (0, 0, 230)
    cv2.putText(strip, label.upper(), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    return strip


def collect_rim_crops(
    video: Path, specs: list[tuple[int, list[float], float, int]], fps: float
) -> dict[int, list[np.ndarray | None]]:
    """For each (shot number, rim box [x1,y1,x2,y2], rim plane y, final crossing frame),
    the rim crops at STACK_TIMES_S around the crossing. One sequential pass over the
    video, which is more reliable than seeking in phone videos."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    want: dict[int, list[tuple[int, int]]] = {}  # frame -> [(shot, slot)]
    boxes = {}
    for num, rim, plane_y, cross in specs:
        boxes[num] = crop_box(rim, plane_y, width, height)
        for slot, dt in enumerate(STACK_TIMES_S):
            fi = min(max(cross + round(dt * fps), 0), max(n_frames - 1, 0))
            want.setdefault(fi, []).append((num, slot))

    got: dict[int, list[np.ndarray | None]] = {num: [None] * len(STACK_TIMES_S) for num, *_ in specs}
    last = max(want) if want else -1
    idx = 0
    while idx <= last:
        if not cap.grab():
            break
        if idx in want:
            _, img = cap.retrieve()
            for num, slot in want[idx]:
                x1, y1, x2, y2 = boxes[num]
                got[num][slot] = img[y1:y2, x1:x2].copy()
        idx += 1
    cap.release()
    return got


def _rows_from_tracks(tracks: dict) -> list[dict]:
    fps = tracks["fps"]
    rows = []
    for s in tracks["shots"]:
        f = s["frames"]
        ref = f.get("release") or f.get("start") or f.get("up") or 0
        rows.append({"shot_number": s["shot_number"], "made": s["made"], "video_offset_s": ref / fps})
    return rows


def export_crops(
    video: Path, tracks_path: Path, truth_path: Path, out: Path, split: str = "train", preview: bool = False
) -> dict[str, int]:
    tracks = json.loads(tracks_path.read_text())
    fps = tracks["fps"]
    shots = {s["shot_number"]: s for s in tracks["shots"]}
    rep = score(_rows_from_tracks(tracks), parse_truth(truth_path.read_text()))
    for n in rep.notes:
        log.warning(n)
    labels = {p.row["shot_number"]: ("make" if p.truth.made else "miss") for p in rep.pairs if p.truth and p.row}

    specs = [(num, shots[num]["rim"], shots[num]["rim_plane_y"], shots[num]["frames"]["cross"]) for num in labels]
    got = collect_rim_crops(video, specs, fps)

    counts = {"make": 0, "miss": 0, "skipped": 0}
    stem = video.stem
    for num, label in sorted(labels.items()):
        crops = got[num]
        if any(c is None or c.size == 0 for c in crops):
            counts["skipped"] += 1
            continue
        d = out / split / label
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{stem}_{num:03d}.jpg"), stack_image(crops), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if preview:
            pd = out / "preview" / split
            pd.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(pd / f"{stem}_{num:03d}_{label}.jpg"), preview_strip(crops, label))
        counts[label] += 1
    return counts
