"""ffmpeg helpers: recording time, per-shot clips, annotated debug video."""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2

from .form import SIDES
from .frames import FrameData, Shot, VideoInfo
from .shots import RimModel, rim_plane

log = logging.getLogger(__name__)

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]


def recorded_at(path: Path) -> datetime:
    """creation_time from the container metadata (phones/GoPros set it), else file mtime."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        ts = json.loads(out).get("format", {}).get("tags", {}).get("creation_time")
        if ts:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception as e:  # noqa: BLE001
        log.debug("ffprobe failed for %s: %s", path, e)
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def make_clip(src: Path, start_s: float, end_s: float, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{max(0.0, start_s):.2f}", "-i", str(src), "-t", f"{end_s - max(0.0, start_s):.2f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.warning("clip failed: %s", r.stderr.strip())
    return r.returncode == 0


def clip_window(s: Shot, fps: float) -> tuple[float, float]:
    start = s.release_frame if s.release_frame is not None else (s.start_frame or s.up_frame)
    return start / fps - 1.5, s.end_frame / fps + 1.0


def write_debug_video(
    info: VideoInfo, frames: list[FrameData], shots: list[Shot], out: Path, make_tolerance: float = 0.6
) -> None:
    """Overlay detections, pose, release point and running tally. Useful for tuning.

    Rim overlay: orange box = raw detection in this frame. Cyan line = the rim
    height the tracker uses (from the smoothed rim), with a green segment for
    the make zone (MAKE_TOLERANCE) and a tick at the rim centre.
    """
    rims = RimModel(frames, info.fps)
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(info.path)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height))
    by_end = {s.end_frame: s for s in shots}
    releases = {s.release_frame: s for s in shots if s.release_frame is not None}
    makes = attempts = 0
    banner, banner_until = "", -1
    for f in frames:
        ok, img = cap.read()
        if not ok:
            break
        if f.rim:
            r = f.rim
            cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 165, 255), 2)
        sr = rims.at(f.idx)
        if sr is not None:
            y = int(round(rim_plane(sr)))
            half = make_tolerance * sr.w / 2
            cv2.line(img, (int(sr.x1), y), (int(sr.x2), y), (255, 255, 0), 1)  # rim height (cyan)
            cv2.line(img, (int(sr.cx - half), y), (int(sr.cx + half), y), (0, 255, 0), 2)  # make zone
            cv2.line(img, (int(sr.cx), y - 6), (int(sr.cx), y + 6), (255, 255, 0), 2)  # rim centre
        if f.ball:
            b = f.ball
            cv2.circle(img, (int(b.cx), int(b.cy)), int(b.w / 2), (255, 128, 0), 2)
        if f.pose is not None:
            k = f.pose.kpts
            for a, c in SKELETON:
                if k[a][2] > 0.3 and k[c][2] > 0.3:
                    cv2.line(img, (int(k[a][0]), int(k[a][1])), (int(k[c][0]), int(k[c][1])), (0, 255, 0), 2)
        if f.idx in releases:
            s = releases[f.idx]
            hand = s.form.get("shooting_hand")
            if f.pose is not None and hand in SIDES:
                w = f.pose.kpts[SIDES[hand]["wrist"]]
                cv2.circle(img, (int(w[0]), int(w[1])), 14, (255, 0, 255), 3)
            elbow = s.form.get("release_elbow_angle")
            banner, banner_until = f"RELEASE  elbow {elbow}", f.idx + int(info.fps)
        if f.idx in by_end:
            s = by_end[f.idx]
            attempts += 1
            makes += int(s.made)
            banner = f"{'MAKE' if s.made else 'MISS'}  arc {s.launch_angle_deg}  entry {s.entry_angle_deg}"
            banner_until = f.idx + int(1.5 * info.fps)
        cv2.putText(img, f"{makes}/{attempts}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
        if f.idx <= banner_until:
            cv2.putText(img, banner, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        vw.write(img)
    cap.release()
    vw.release()
