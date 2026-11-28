import json
import os
import time

import pytest

from tracker import evaluate as ev
from tracker.config import Config
from tracker.main import main


def row(n, made, off=None, exit_=None, prob=None, arc=50.0):
    return {"shot_number": n, "made": made, "video_offset_s": 10.0 * n,
            "rim_offset": off if off is not None else (0.1 if made else 2.0),
            "exit_speed_mps": exit_, "make_prob": prob, "launch_angle_deg": arc, "release_elbow_angle": 150.0}


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.video_root = tmp_path
    c.debug.mkdir()
    c.processed.mkdir()
    c.make_tolerance, c.max_make_exit_speed = 0.6, 3.0
    c.make_classifier, c.make_classifier_threshold, c.make_classifier_rescue = None, 0.7, 0.95
    return c


def _session(cfg, stem, rows, truth=None):
    (cfg.debug / f"{stem}.json").write_text(json.dumps(rows))
    if truth is not None:
        (cfg.debug / f"{stem}_truth.txt").write_text(truth)


def test_eval_table_from_json_results(cfg, tmp_path):
    _session(cfg, "a", [row(1, True), row(2, False), row(3, True, exit_=4.0, prob=1.0)], "make miss make")
    _session(cfg, "b", [row(1, True), row(2, True)], "make miss")
    _session(cfg, "c", [row(1, True)])  # no truth file: ignored without a database
    st = ev.Settings(0.6, 3.0, 0.7, 0.95, 0.9)
    text = ev.run(cfg, st, use_db=False, dataset=tmp_path / "ds")
    lines = {ln.split()[0]: ln for ln in text.splitlines() if ln and ln[0] in "abA"}
    assert "3/3 100.0%" in lines["a.json"]  # #3: exit-speed miss undone by a confident classifier
    assert "1/2  50.0%" in lines["b.json"] and lines["b.json"].split()[4:6] == ["1", "0"]  # one false make
    assert "4/5  80.0%" in lines["ALL"]
    assert "max_exit_speed=3" in text and "rescue=0.95" in text


def test_eval_replays_other_settings(cfg, tmp_path):
    _session(cfg, "a", [row(1, True), row(2, False), row(3, True, exit_=4.0, prob=1.0)], "make miss make")
    st = ev.Settings(0.6, 3.0, None, None, 0.9)  # classifier off: the exit-speed rule stands
    assert "2/3" in ev.run(cfg, st, use_db=False, dataset=tmp_path / "ds")


def test_details_lists_problem_shots(cfg, tmp_path):
    _session(cfg, "b", [row(1, True), row(2, True)], "make miss")
    text = ev.run(cfg, ev.Settings(0.6, None, None, None, 0.9), details=True, use_db=False, dataset=tmp_path / "ds")
    assert "== b.json" in text and "app MAKE, actual miss" in text


def _crop(ds, stem, n, label="make", t=None):
    p = ds / "train" / label / f"{stem}_{n:03d}.jpg"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    if t:
        os.utime(p, (t, t))
    return p


def test_trained_on_needs_a_model_newer_than_the_images(tmp_path):
    ds, model = tmp_path / "ds", tmp_path / "make_cls.pt"
    assert not ev.trained_on("s", ds, str(model))
    _crop(ds, "s", 1, t=time.time() - 100)
    _crop(ds, "s_2026-01-01_1200", 1)  # longer name starting with the same stem: not "s"
    model.write_bytes(b"m")
    assert ev.trained_on("s", ds, str(model))
    _crop(ds, "s", 2, "miss", t=time.time() + 100)  # added after the model was trained
    assert not ev.trained_on("s", ds, str(model))


def test_refresh_crops_only_when_missing_or_truth_changed(cfg, tmp_path, monkeypatch):
    ds = tmp_path / "ds"
    calls = []

    def fake_export(video, tracks, truth, out, split, preview=False):
        calls.append(video.name)
        _crop(out, video.stem, 1)
        return {"make": 1, "miss": 0, "skipped": 0}

    import tracker.crops
    monkeypatch.setattr(tracker.crops, "export_crops", fake_export)
    _session(cfg, "s", [row(1, True)], "make")
    (cfg.processed / "s.mov").write_bytes(b"v")
    (cfg.debug / "s_tracks.json").write_text("{}")
    r = ev.Result(ev.Session("s.mov", []), cfg.debug / "s_truth.txt", None, {}, False)

    assert ev.refresh_crops(r, cfg, ds).startswith("exported")
    assert ev.refresh_crops(r, cfg, ds) == ""  # up to date
    stale = _crop(ds, "s", 7, "miss", t=time.time() - 100)  # an old image from a wrong truth line
    for p in (ds / "train" / "make").iterdir():
        os.utime(p, (time.time() - 100,) * 2)
    os.utime(r.truth_path, None)  # truth corrected
    assert ev.refresh_crops(r, cfg, ds).startswith("re-exported")
    assert not stale.exists() and calls == ["s.mov", "s.mov"]

    r2 = ev.Result(ev.Session("gone.mov", []), cfg.debug / "s_truth.txt", None, {}, False)
    assert "no crops" in ev.refresh_crops(r2, cfg, ds)


def test_cli(cfg, capsys, monkeypatch):
    monkeypatch.setenv("VIDEO_ROOT", str(cfg.video_root))
    monkeypatch.setenv("MAKE_CLASSIFIER", "")
    _session(cfg, "a", [row(1, True, exit_=4.0), row(2, False)], "make miss")
    main(["eval", "--no-db", "--max-exit-speed", "3"])
    assert "1/2" in capsys.readouterr().out
    main(["eval", "--no-db", "--max-exit-speed", "off"])
    out = capsys.readouterr().out
    assert "2/2" in out and "max_exit_speed=off" in out
