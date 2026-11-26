"""Entry point.

  python -m tracker.main watch                     # poll VIDEO_ROOT/incoming forever
  python -m tracker.main process clip.mp4 --debug  # one file; --no-db prints JSON
  python -m tracker.main score results.json truth.txt   # accuracy vs your own make/miss list
  python -m tracker.main crops video.mov tracks.json truth.txt --out DIR --split train
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path

from .config import Config
from .pipeline import analyze_frames, dump_tracks, shot_row

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}
log = logging.getLogger("tracker")


def process_file(
    path: Path, cfg: Config, detector, *, save: bool = True, debug: bool = False, tracks: bool = False
) -> list[dict]:
    from . import media

    t0 = time.time()
    info, frames = detector.run(
        str(path), progress=lambda i, n: log.info("  %s: frame %d/%d", path.name, i, n)
    )
    shots = analyze_frames(frames, info.fps, cfg)
    rows = [shot_row(s, info.fps) for s in shots]

    if cfg.make_clips and save:
        for s, row in zip(shots, rows):
            start, end = media.clip_window(s, info.fps)
            rel = Path("clips") / path.stem / f"{s.number:03d}_{'make' if s.made else 'miss'}.mp4"
            if media.make_clip(path, start, end, cfg.video_root / rel):
                row["clip_path"] = str(rel)

    if tracks:
        base = cfg.debug if cfg.video_root.exists() else path.parent / "debug"
        base.mkdir(parents=True, exist_ok=True)
        out = base / f"{path.stem}_tracks.json"
        out.write_text(json.dumps(dump_tracks(frames, info.fps, shots)))
        log.info("tracks -> %s", out)

    if debug or cfg.debug_video:
        # never write next to the input: a debug file in incoming/ would be picked up as a new session
        base = cfg.debug if cfg.video_root.exists() else path.parent / "debug"
        out = base / f"{path.stem}_debug.mp4"
        media.write_debug_video(info, frames, shots, out, cfg.make_tolerance)
        log.info("debug video -> %s", out)

    makes = sum(r["made"] for r in rows)
    log.info("%s: %d/%d makes, %.0fs video in %.0fs", path.name, makes, len(rows), info.duration_s, time.time() - t0)

    if save:
        from .db import save_session

        sid = save_session(cfg.database_url, info, path.name, media.recorded_at(path), rows)
        log.info("saved session %d", sid)
    return rows


def _stable(path: Path, sizes: dict[Path, int]) -> bool:
    """A file is ready when its size hasn't changed since the last poll (upload finished)."""
    size = path.stat().st_size
    ready = sizes.get(path) == size and size > 0
    sizes[path] = size
    return ready


def watch(cfg: Config) -> None:
    from .detect import Detector

    for d in (cfg.incoming, cfg.processed, cfg.failed, cfg.clips):
        d.mkdir(parents=True, exist_ok=True)
    detector = Detector(cfg)
    sizes: dict[Path, int] = {}
    log.info("watching %s every %.0fs", cfg.incoming, cfg.poll_seconds)
    # Polling instead of inotify: inotify doesn't fire on NFS/SMB mounts from a NAS.
    while True:
        for path in sorted(cfg.incoming.iterdir()):
            if (
                path.suffix.lower() not in VIDEO_EXT
                or path.stem.endswith("_debug")
                or path.name.startswith(".")  # in-progress uploads (Synology Drive, rsync)
                or not _stable(path, sizes)
            ):
                continue
            sizes.pop(path, None)
            log.info("processing %s", path.name)
            try:
                process_file(path, cfg, detector)
                shutil.move(str(path), cfg.processed / path.name)
            except Exception:  # noqa: BLE001 - keep the watcher alive
                log.error("failed on %s", path.name, exc_info=True)
                shutil.move(str(path), cfg.failed / path.name)
                (cfg.failed / f"{path.name}.log").write_text(traceback.format_exc())
        time.sleep(cfg.poll_seconds)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tracker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("watch")
    p = sub.add_parser("process")
    p.add_argument("video", type=Path)
    p.add_argument("--debug", action="store_true", help="write an annotated video")
    p.add_argument("--no-db", action="store_true", help="print results as JSON instead of saving")
    p.add_argument("--dump-tracks", action="store_true", help="save per-shot ball positions to debug/<name>_tracks.json")
    sc = sub.add_parser("score", help="compare results with your own make/miss list")
    sc.add_argument("results", type=Path, help="JSON from 'process --no-db'")
    sc.add_argument("truth", type=Path, help="make/miss list (see tracker/score.py for formats)")
    sc.add_argument("--make-tolerance", type=float, help="re-score with this MAKE_TOLERANCE")
    sc.add_argument("--max-exit-speed", type=float, help="re-score with this MAX_MAKE_EXIT_SPEED (m/s)")
    sc.add_argument("--window", type=float, default=2.5, help="seconds allowed between a timed truth shot and a detection")
    cr = sub.add_parser("crops", help="export labeled rim images for the make/miss classifier")
    cr.add_argument("video", type=Path)
    cr.add_argument("tracks", type=Path, help="JSON from 'process --dump-tracks'")
    cr.add_argument("truth", type=Path, help="make/miss list (same formats as score)")
    cr.add_argument("--out", type=Path, required=True, help="dataset folder, e.g. /models/datasets/makecls")
    cr.add_argument("--split", default="train", choices=["train", "val"])
    cr.add_argument("--preview", action="store_true", help="also write viewable strips to OUT/preview/")
    args = ap.parse_args(argv)

    if args.cmd == "crops":
        from .crops import export_crops

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        c = export_crops(args.video, args.tracks, args.truth, args.out, args.split, args.preview)
        print(f"{args.video.name} -> {args.out / args.split}: {c['make']} make, {c['miss']} miss"
              + (f", {c['skipped']} skipped (crop outside video)" if c["skipped"] else ""))
        return

    if args.cmd == "score":
        from .score import format_report, load_rows, parse_truth, rescore, score

        rows = rescore(load_rows(args.results), args.make_tolerance, args.max_exit_speed)
        settings = ", ".join(
            f"{k}={v}" for k, v in (("make_tolerance", args.make_tolerance), ("max_exit_speed", args.max_exit_speed))
            if v is not None
        )
        title = f"{args.results.name} vs {args.truth.name}" + (f"  (re-scored: {settings})" if settings else "")
        print(format_report(score(rows, parse_truth(args.truth.read_text()), args.window), title))
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config()
    if args.cmd == "watch":
        watch(cfg)
    else:
        from .detect import Detector

        if args.no_db:
            cfg.make_clips = False
        rows = process_file(
            args.video, cfg, Detector(cfg), save=not args.no_db, debug=args.debug, tracks=args.dump_tracks
        )
        if args.no_db:
            json.dump(rows, sys.stdout, indent=2, default=str)
            print()


if __name__ == "__main__":
    main()
