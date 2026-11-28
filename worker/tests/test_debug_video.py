import cv2
import numpy as np

from synth import RIM, shot_frames
from tracker.config import Config
from tracker.frames import VideoInfo
from tracker.media import write_debug_video
from tracker.pipeline import analyze_frames

FPS = 60.0


def _render(tmp_path, frames):
    src = tmp_path / "src.mp4"
    vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
    for _ in frames:
        vw.write(np.zeros((720, 1280, 3), np.uint8))
    vw.release()
    cfg = Config()
    cfg.shooting_hand = "auto"
    shots = analyze_frames(frames, FPS, cfg)
    out = tmp_path / "debug.mp4"
    write_debug_video(VideoInfo(str(src), FPS, len(frames), 1280, 720), frames, shots, out)
    cap = cv2.VideoCapture(str(out))
    imgs = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        imgs.append(img)
    return shots, imgs


def _pixels(img, bgr):
    return int((np.abs(img.astype(int) - bgr).sum(axis=2) < 60).sum())


def test_measured_arc_is_drawn_in_yellow(tmp_path):
    frames, rel = shot_frames(target_x=RIM.cx)
    shots, imgs = _render(tmp_path, frames)
    s = shots[0]
    assert s.launch_angle_deg is not None and s.arc_fit["ok"]
    mid = imgs[(s.arc_fit["start"] + s.arc_fit["end"]) // 2]
    assert _pixels(mid, (0, 255, 255)) > 300  # the curve (and banner text later)
    assert _pixels(imgs[rel - 5], (0, 255, 255)) == 0  # not before the release


def test_skipped_arc_is_drawn_in_red_with_the_reason(tmp_path):
    frames, _ = shot_frames(angle=77, target_x=RIM.cx)
    shots, imgs = _render(tmp_path, frames)
    s = shots[0]
    assert s.launch_angle_deg is None and not s.arc_fit["ok"] and "implausible" in s.arc_note
    mid = imgs[(s.arc_fit["start"] + s.arc_fit["end"]) // 2]
    assert _pixels(mid, (0, 0, 255)) > 300
