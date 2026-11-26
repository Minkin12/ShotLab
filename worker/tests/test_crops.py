import json

import cv2
import numpy as np

from tracker.crops import IMG_SIZE, crop_box, stack_image
from tracker.main import main


def test_crop_box_is_square_and_inside_frame():
    x1, y1, x2, y2 = crop_box([900, 300, 1000, 420], 315, 1920, 1080)
    assert x2 - x1 == y2 - y1 == 250
    assert x1 == 825 and y1 == 215  # centred on the rim, 40% above the rim plane
    x1, y1, x2, y2 = crop_box([1850, 10, 1910, 60], 20, 1920, 1080)  # rim at the corner
    assert 0 <= x1 and x2 <= 1920 and 0 <= y1 and y2 <= 1080


def test_stack_puts_motion_in_colour():
    still = np.full((100, 100, 3), 120, np.uint8)
    moved = still.copy()
    moved[40:60, 40:60] = 255  # something appears in the middle at the latest moment
    img = stack_image([still, still, moved])
    assert img.shape == (IMG_SIZE, IMG_SIZE, 3)
    corner, centre = img[5, 5], img[IMG_SIZE // 2, IMG_SIZE // 2]
    assert corner[0] == corner[1] == corner[2]  # static background stays gray
    assert centre[0] > centre[2]  # blue (latest) brighter than red (earliest) where it moved


def test_crops_command_end_to_end(tmp_path):
    fps, w, h = 30.0, 640, 360
    video = tmp_path / "s.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(120):
        img = np.full((h, w, 3), 90, np.uint8)
        cv2.circle(img, (400, 40 + 3 * i), 10, (40, 80, 200), -1)  # a ball falling past the rim
        vw.write(img)
    vw.release()
    rim = [370, 100, 430, 160]
    shots = [
        {"shot_number": n, "made": made, "rim": rim, "rim_plane_y": 109,
         "frames": {"release": c - 30, "start": c - 30, "up": c - 10, "first_cross": c, "cross": c, "end": c + 5}}
        for n, made, c in [(1, True, 20), (2, True, 50), (3, False, 80)]
    ]
    tracks = tmp_path / "s_tracks.json"
    tracks.write_text(json.dumps({"fps": fps, "shots": shots}))
    truth = tmp_path / "truth.txt"
    truth.write_text("make miss miss\n")  # shot 2 was really a miss
    out = tmp_path / "ds"
    main(["crops", str(video), str(tracks), str(truth), "--out", str(out), "--split", "val", "--preview"])
    assert sorted(p.name for p in (out / "val" / "make").glob("*.jpg")) == ["s_001.jpg"]
    assert sorted(p.name for p in (out / "val" / "miss").glob("*.jpg")) == ["s_002.jpg", "s_003.jpg"]
    assert cv2.imread(str(out / "val" / "make" / "s_001.jpg")).shape == (IMG_SIZE, IMG_SIZE, 3)
    assert len(list((out / "preview" / "val").glob("*.jpg"))) == 3
