import math

import numpy as np
import pytest

from synth import BALL_W, RIM, SCALE, flight_speed, shot_frames
from tracker.config import Config
from tracker.form import joint_angle
from tracker.frames import Box, FrameData
from tracker.pipeline import analyze_frames, shot_row
from tracker.shots import clean_track, rim_plane

FPS = 60.0


@pytest.fixture
def cfg():
    c = Config()
    c.shooting_hand = "auto"
    return c


def test_joint_angle():
    a, b = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    assert joint_angle(a, b, np.array([2.0, 0.0])) == pytest.approx(180)
    assert joint_angle(a, b, np.array([1.0, 1.0])) == pytest.approx(90)
    assert joint_angle(a, b, np.array([0.0, 0.0])) == pytest.approx(0)


def test_make(cfg):
    frames, _ = shot_frames(target_x=RIM.cx)
    shots = analyze_frames(frames, FPS, cfg)
    assert len(shots) == 1
    assert shots[0].made


def test_miss_long(cfg):
    frames, _ = shot_frames(target_x=RIM.cx + 50)  # crosses rim plane past the rim
    shots = analyze_frames(frames, FPS, cfg)
    assert len(shots) == 1
    assert not shots[0].made


def test_flight_metrics(cfg):
    frames, _ = shot_frames(angle=50, target_x=RIM.cx)
    s = analyze_frames(frames, FPS, cfg)[0]
    assert s.launch_angle_deg == pytest.approx(50, abs=1.5)
    v_true = flight_speed(50, (RIM.cx, RIM.cy)) * SCALE
    assert s.release_speed_mps == pytest.approx(v_true, rel=0.05)
    assert s.entry_angle_deg > 0
    assert s.apex_above_rim_m > 0


def test_release_and_form(cfg):
    frames, rel = shot_frames(knee_dip_before=0.4)
    s = analyze_frames(frames, FPS, cfg)[0]
    assert s.form["shooting_hand"] == "right"
    assert abs(s.release_frame - rel) <= 2
    assert s.form["min_knee_angle"] == pytest.approx(140, abs=3)
    assert s.form["dip_to_release_ms"] == pytest.approx(400, abs=40)
    assert 0 < s.form["release_elbow_angle"] <= 180
    # synthetic wrist y=462, nose y=450, ankle y=780 -> (780-462)/(780-450)
    assert s.form["release_height_ratio"] == pytest.approx(0.96, abs=0.02)
    assert s.form["torso_lean_deg"] == pytest.approx(0, abs=1)
    assert s.form["jump_ratio"] == pytest.approx(0, abs=0.01)  # set shot
    row = shot_row(s, FPS)
    assert row["release_elbow_angle"] == s.form["release_elbow_angle"]
    assert row["video_offset_s"] == pytest.approx(rel / FPS, abs=0.05)


def test_no_pose_still_tracks(cfg):
    frames, rel = shot_frames(with_pose=False)
    s = analyze_frames(frames, FPS, cfg)[0]
    assert s.made
    assert s.release_frame is None
    # release found from the ball's flight alone
    assert abs(s.start_frame - rel) <= 2
    assert s.launch_angle_deg == pytest.approx(50, abs=2)
    assert shot_row(s, FPS)["release_elbow_angle"] is None


def test_two_shots(cfg):
    a, _ = shot_frames(target_x=RIM.cx)
    b, _ = shot_frames(target_x=RIM.cx + 50, start=len(a) + 30)
    shots = analyze_frames(a + b, FPS, cfg)
    assert [s.made for s in shots] == [True, False]
    assert [s.number for s in shots] == [1, 2]


def test_clean_track_drops_outlier():
    frames = []
    for i in range(10):
        x = 100 + 10 * i
        if i == 5:
            x = 2000  # spurious detection
        frames.append(FrameData(i, i / FPS, ball=Box(x - 15, 300, x + 15, 330)))
    xs = [p[1] for p in clean_track(frames, FPS)]
    assert 2000 not in xs
    assert len(xs) == 9


def test_ball_held_overhead_is_not_a_shot(cfg):
    frames = []
    for i in range(int(5 * FPS)):  # ball sits above the rim for 5 s, never drops
        frames.append(FrameData(i, i / FPS, ball=Box(985, 200, 985 + BALL_W, 230), rim=RIM))
    assert analyze_frames(frames, FPS, cfg) == []


@pytest.mark.parametrize("fps", [30.0, 60.0])
def test_noisy_detections(cfg, fps):
    """Jittery boxes and ~20% dropped ball detections, like a real detector."""
    rng = np.random.default_rng(7)
    frames, rel = shot_frames(fps=fps, target_x=RIM.cx)
    for f in frames:
        if f.ball is not None:
            if rng.random() < 0.2:
                f.ball = None
                continue
            j = rng.normal(0, 1.5, 2)
            f.ball = Box(f.ball.x1 + j[0], f.ball.y1 + j[1], f.ball.x2 + j[0], f.ball.y2 + j[1])
        if f.pose is not None:
            f.pose.kpts[:, :2] += rng.normal(0, 2.0, (17, 2))
    shots = analyze_frames(frames, fps, cfg)
    assert len(shots) == 1 and shots[0].made
    s = shots[0]
    assert s.launch_angle_deg == pytest.approx(50, abs=3)
    assert s.form["min_knee_angle"] == pytest.approx(140, abs=6)


def test_rim_box_including_net(cfg):
    """Datasets often label rim+net as one tall box; the rim plane must stay at the top."""
    tall = Box(RIM.x1, RIM.y1, RIM.x2, RIM.y1 + 75)  # 60 wide, 75 tall (net hanging below)
    frames, _ = shot_frames(target_x=RIM.cx)
    for f in frames:
        f.rim = tall
    shots = analyze_frames(frames, FPS, cfg)
    assert len(shots) == 1 and shots[0].made
    assert shots[0].launch_angle_deg == pytest.approx(50, abs=2)


def test_release_when_ball_lost_during_extension(cfg):
    """Ball detections drop out while the arm extends (motion blur): release must
    still land at full extension, not at the set point where the ball was last seen."""
    frames, rel = shot_frames()
    shoulder, elbow = np.array([390.0, 500.0]), np.array([430.0, 500.0])
    by_idx = {f.idx: f for f in frames}
    for i in range(rel - 15, rel + 6):
        theta = math.radians(min(175.0, 60.0 + (175.0 - 60.0) * (i - (rel - 15)) / 15))
        wrist = elbow + 40 * np.array([-math.cos(theta), -math.sin(theta)])
        k = by_idx[i].pose.kpts
        k[6, :2], k[8, :2], k[10, :2] = shoulder, elbow, wrist
        if i <= rel - 9:  # ball still seen in the hand at the set point
            by_idx[i].ball = Box(wrist[0] - 15, wrist[1] - 15, wrist[0] + 15, wrist[1] + 15)
        elif i <= rel:  # ...then lost during the extension
            by_idx[i].ball = None
    s = analyze_frames(frames, FPS, cfg)[0]
    assert abs(s.release_frame - rel) <= 2
    assert s.form["release_elbow_angle"] > 160


def test_sparse_flight_gives_no_arc(cfg):
    """Only a handful of in-flight detections: the shot is still counted, arc stays empty."""
    frames, rel = shot_frames(target_x=RIM.cx)
    for f in frames:
        if f.ball is not None and f.idx > rel + 3 and (f.idx - rel) % 20:
            f.ball = None  # keep every 20th frame in flight only (~6 points)
    shots = analyze_frames(frames, FPS, cfg)
    assert len(shots) == 1
    assert shots[0].launch_angle_deg is None and shots[0].release_speed_mps is None


def test_ball_lost_around_apex_gives_no_arc(cfg):
    frames, rel = shot_frames(target_x=RIM.cx)
    for f in frames:
        if f.ball is not None and rel + 10 <= f.idx <= rel + 45:
            f.ball = None  # long hole in the middle of the flight
    s = analyze_frames(frames, FPS, cfg)[0]
    assert s.launch_angle_deg is None


def test_outlier_points_give_no_arc(cfg):
    rng = np.random.default_rng(3)
    frames, rel = shot_frames(target_x=RIM.cx)
    for f in frames:
        if f.ball is not None and f.idx > rel and rng.random() < 0.5:
            j = rng.normal(0, 40, 2)  # wild jitter: another object mistaken for the ball
            f.ball = Box(f.ball.x1 + j[0], f.ball.y1 + j[1], f.ball.x2 + j[0], f.ball.y2 + j[1])
    s = analyze_frames(frames, FPS, cfg)[0]
    assert s.launch_angle_deg is None


def test_rim_bounce_keeps_arc_and_scores_miss(cfg):
    """Ball reaches the rim, bounces up and drops off the far side: the arc is measured
    from the free flight only, and the last crossing (outside the rim) makes it a miss."""
    frames, rel = shot_frames(target_x=RIM.x2 - 5)  # arrives on the back of the rim
    plane = rim_plane(RIM)
    balls = [f for f in frames if f.idx > rel and f.ball]
    hit = next(b.idx for a, b in zip(balls, balls[1:]) if a.ball.cy <= plane < b.ball.cy)
    by_idx = {f.idx: f for f in frames}
    bx = RIM.x2 + 25  # bounces off behind the rim
    for k, i in enumerate(range(hit, hit + 90)):
        if i not in by_idx:
            break
        y = plane - 60 * math.sin(math.pi * k / 30) if k < 30 else plane + 12 * (k - 30)
        by_idx[i].ball = None if y > 900 else Box(bx - 15, y - 15, bx + 15, y + 15)
    s = analyze_frames(frames, FPS, cfg)[0]
    assert not s.made
    assert s.launch_angle_deg == pytest.approx(50, abs=2)


def test_few_stray_detections_are_trimmed(cfg):
    frames, rel = shot_frames(target_x=RIM.cx)
    flight = [f for f in frames if f.ball is not None and rel < f.idx < rel + 50]
    for f in flight[5:30:8]:  # 3-4 detections knocked well off the arc
        f.ball = Box(f.ball.x1 + 45, f.ball.y1 - 45, f.ball.x2 + 45, f.ball.y2 - 45)
    s = analyze_frames(frames, FPS, cfg)[0]
    assert s.launch_angle_deg == pytest.approx(50, abs=2)


def test_implausible_arc_is_skipped(cfg):
    frames, _ = shot_frames(angle=77, target_x=RIM.cx)  # steeper than any real jump shot
    shots = analyze_frames(frames, FPS, cfg)
    assert len(shots) == 1
    assert shots[0].launch_angle_deg is None


def test_trimming_keeps_the_only_points_after_the_apex(cfg):
    """Ball mostly lost on the way down: the couple of descending detections left must
    not be thrown away as outliers, or the arc can't be measured at all."""
    frames, rel = shot_frames(target_x=RIM.cx)
    balls = [f for f in frames if f.ball is not None and f.idx > rel]
    apex = min(balls, key=lambda f: f.ball.cy).idx
    descending = [f for f in balls if apex + 3 < f.idx and f.ball.cy < rim_plane(RIM) - 20]
    keep = {descending[len(descending) // 3].idx, descending[2 * len(descending) // 3].idx}
    for f in descending:
        if f.idx not in keep:
            f.ball = None
    f = next(f for f in frames if f.idx == max(keep))
    f.ball = Box(f.ball.x1 + 33, f.ball.y1, f.ball.x2 + 33, f.ball.y2)  # ~1.1 ball widths off
    s = analyze_frames(frames, FPS, cfg)[0]
    assert s.launch_angle_deg == pytest.approx(50, abs=3)


def test_dump_tracks_is_json_ready(cfg):
    import json

    from tracker.pipeline import dump_tracks

    frames, _ = shot_frames(target_x=RIM.cx)
    shots = analyze_frames(frames, FPS, cfg)
    d = json.loads(json.dumps(dump_tracks(frames, FPS, shots)))
    s = d["shots"][0]
    assert d["fps"] == FPS and s["made"] is True
    assert s["frames"]["cross"] >= s["frames"]["first_cross"] >= s["frames"]["up"]
    assert len(s["ball"]) > 50 and len(s["ball"][0]) == 6


def _slow_through_net(frames, speed_px_s=100.0):
    """After the ball first drops below the rim height, make it sink slowly (net drag)."""
    plane = rim_plane(RIM)
    balls = [f for f in frames if f.ball is not None]
    c = next(b.idx for a, b in zip(balls, balls[1:]) if a.ball.cy <= plane < b.ball.cy and b.ball.cy > 0)
    x0, y0 = next(f.ball.cx for f in frames if f.idx == c), next(f.ball.cy for f in frames if f.idx == c)
    for f in frames:
        if f.idx >= c:
            y = y0 + speed_px_s * (f.idx - c) / FPS
            f.ball = Box(x0 - 15, y - 15, x0 + 15, y + 15)


def test_exit_speed_is_measured_and_saved(cfg):
    frames, _ = shot_frames(target_x=RIM.cx)  # synthetic ball falls freely through the "net"
    s = analyze_frames(frames, FPS, cfg)[0]
    assert s.exit_speed_mps is not None and s.exit_speed_mps > 3
    assert shot_row(s, FPS)["exit_speed_mps"] == s.exit_speed_mps


def test_fast_exit_overrules_make_only_when_enabled(cfg):
    frames, _ = shot_frames(target_x=RIM.cx)
    assert analyze_frames(frames, FPS, cfg)[0].made  # off by default
    cfg.max_make_exit_speed = 1.95
    assert not analyze_frames(frames, FPS, cfg)[0].made  # free fall past the rim -> miss


def test_slow_exit_stays_a_make(cfg):
    frames, _ = shot_frames(target_x=RIM.cx)
    _slow_through_net(frames, speed_px_s=100.0)  # ~0.76 m/s at this scale
    cfg.max_make_exit_speed = 1.95
    s = analyze_frames(frames, FPS, cfg)[0]
    assert s.made
    assert s.exit_speed_mps == pytest.approx(100 * SCALE, abs=0.05)


def test_free_flight_ends_after_the_apex_with_tall_rim_box(cfg):
    """Real models box rim+net and see a big ball: the ball passes 'near rim height' on
    the way UP. The free flight must still end on the way down, or the arc can't be fit."""
    tall = Box(RIM.x1, RIM.y1, RIM.x2, RIM.y1 + 75)
    frames, _ = shot_frames(target_x=RIM.cx)
    for f in frames:
        f.rim = tall
        if f.ball is not None:  # ball twice as big on screen
            f.ball = Box(f.ball.cx - 30, f.ball.cy - 30, f.ball.cx + 30, f.ball.cy + 30)
    s = analyze_frames(frames, FPS, cfg)[0]
    apex = min((f for f in frames if f.ball and s.up_frame <= f.idx <= s.cross_frame), key=lambda f: f.ball.cy).idx
    assert s.first_cross_frame > apex
    # measured at all is the point; the doubled ball makes the synthetic release a bit late
    assert s.launch_angle_deg == pytest.approx(50, abs=5)
