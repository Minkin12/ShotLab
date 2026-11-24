"""Configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: str) -> list[str]:
    return [s.strip().lower() for s in _env(name, default).split(",") if s.strip()]


@dataclass
class Config:
    # models
    object_model: str = field(default_factory=lambda: _env("OBJECT_MODEL", "/models/ball_rim.pt"))
    pose_model: str = field(default_factory=lambda: _env("POSE_MODEL", "yolo11s-pose.pt"))
    ball_classes: list[str] = field(default_factory=lambda: _list("BALL_CLASSES", "ball,basketball,sports ball"))
    rim_classes: list[str] = field(default_factory=lambda: _list("RIM_CLASSES", "rim,hoop,basketball hoop"))
    ball_conf: float = field(default_factory=lambda: float(_env("BALL_CONF", "0.30")))
    rim_conf: float = field(default_factory=lambda: float(_env("RIM_CONF", "0.40")))
    pose_conf: float = field(default_factory=lambda: float(_env("POSE_CONF", "0.40")))
    device: str = field(default_factory=lambda: _env("DEVICE", "0"))  # "0" = first GPU, "cpu"
    imgsz: int = field(default_factory=lambda: int(_env("IMGSZ", "960")))

    # shot logic
    shooting_hand: str = field(default_factory=lambda: _env("SHOOTING_HAND", "auto"))  # auto|right|left
    make_tolerance: float = field(default_factory=lambda: float(_env("MAKE_TOLERANCE", "0.7")))
    max_flight_s: float = field(default_factory=lambda: float(_env("MAX_FLIGHT_S", "3.0")))
    shot_cooldown_s: float = field(default_factory=lambda: float(_env("SHOT_COOLDOWN_S", "0.5")))

    # files
    video_root: Path = field(default_factory=lambda: Path(_env("VIDEO_ROOT", "/videos")))
    poll_seconds: float = field(default_factory=lambda: float(_env("POLL_SECONDS", "15")))
    make_clips: bool = field(default_factory=lambda: _bool("MAKE_CLIPS", True))
    debug_video: bool = field(default_factory=lambda: _bool("DEBUG_VIDEO", False))

    # database
    database_url: str = field(
        default_factory=lambda: _env("DATABASE_URL", "postgresql://shots:shots@db:5432/shots")
    )

    @property
    def incoming(self) -> Path:
        return self.video_root / "incoming"

    @property
    def processed(self) -> Path:
        return self.video_root / "processed"

    @property
    def failed(self) -> Path:
        return self.video_root / "failed"

    @property
    def clips(self) -> Path:
        return self.video_root / "clips"

    @property
    def debug(self) -> Path:
        return self.video_root / "debug"
