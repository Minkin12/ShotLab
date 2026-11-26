import json

import pytest

from tracker.main import main
from tracker.score import format_report, parse_truth, rescore, score


def row(n, made, t=None, off=None, exit_=None, arc=None):
    return {
        "shot_number": n, "made": made, "video_offset_s": t if t is not None else 10.0 * n,
        "rim_offset": off if off is not None else (0.1 if made else 2.0),
        "exit_speed_mps": exit_, "launch_angle_deg": arc, "release_elbow_angle": 150.0,
    }


def test_parse_truth_formats():
    assert [s.made for s in parse_truth("miss make\nMAKE, miss # comment\n1 0 y n o x")] == [
        False, True, True, False, True, False, True, False, True, False]
    timed = parse_truth("0:12.5 make\n73 miss\n1:30 1\n")
    assert [(s.t, s.made) for s in timed] == [(12.5, True), (73.0, False), (90.0, True)]
    with pytest.raises(ValueError):
        parse_truth("make maybe")


def test_equal_counts_line_up_one_to_one():
    rows = [row(1, True), row(2, True), row(3, False)]
    rep = score(rows, parse_truth("miss make miss"))
    text = format_report(rep)
    assert "2/3 correct" in text and "false makes:    1  #1" in text
    assert "matched by order" in text and "Note" not in text


def test_missed_shot_found_by_order():
    rows = [row(1, False), row(2, True), row(3, True)]  # the real 3rd shot (a miss) wasn't detected
    rep = score(rows, parse_truth("miss make make miss"))
    assert sum(p.row is None for p in rep.pairs) == 1
    text = format_report(rep)
    assert "1 missed, 0 extra" in text and "3/3 correct" in text and "Note:" in text


def test_timed_truth_matches_by_time_and_finds_extras():
    rows = [row(1, True, t=12.0), row(2, False, t=40.0), row(3, True, t=80.0)]
    rep = score(rows, parse_truth("0:12.5 make\n1:20 miss\n2:00 make"))
    text = format_report(rep)
    assert "1 missed, 1 extra" in text  # 2:00 not detected; detection at 40 s isn't a real shot
    assert "false makes:    1  #3" in text and "matched by time" in text


def test_rescore_with_new_settings():
    rows = [row(1, True, off=0.65, exit_=1.0), row(2, True, off=0.1, exit_=2.5), row(3, False, off=0.58)]
    new = rescore(rows, make_tolerance=0.6, max_exit_speed=2.0)
    assert [r["made"] for r in new] == [False, False, True]
    assert [r["made"] for r in rows] == [True, True, False]  # originals untouched


def test_exit_speed_verdict():
    rows = [row(1, True, exit_=1.1), row(2, True, exit_=1.8), row(3, True, exit_=2.3), row(4, True, exit_=2.9)]
    text = format_report(score(rows, parse_truth("make make miss miss")))
    assert "separable" in text and "~2.05 m/s" in text
    rows[1]["exit_speed_mps"] = 2.6
    assert "overlap" in format_report(score(rows, parse_truth("make make miss miss")))


def test_cli(tmp_path, capsys):
    res, tru = tmp_path / "s.json", tmp_path / "truth.txt"
    res.write_text("INFO some log line\n" + json.dumps([row(1, True), row(2, False)]))
    tru.write_text("make miss\n")
    main(["score", str(res), str(tru)])
    out = capsys.readouterr().out
    assert "2/2 correct (100%)" in out and "s.json vs truth.txt" in out