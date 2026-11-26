import cv2
import numpy as np

from tracker.classify import apply_classifier
from tracker.crops import IMG_SIZE
from tracker.frames import Box, Shot

FPS = 30.0


class FakeClassifier:
    """Returns preset p(miss) values in shot order and records what it was shown."""

    def __init__(self, p):
        self.p, self.seen = p, []

    def p_miss(self, images):
        self.seen += images
        return self.p[: len(images)]


def _video(tmp_path):
    path = tmp_path / "s.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 360))
    for _ in range(150):
        vw.write(np.full((360, 640, 3), 90, np.uint8))
    vw.release()
    return path


def _shot(n, made, cross):
    rim = Box(370, 100, 430, 160)
    return Shot(number=n, up_frame=cross - 10, cross_frame=cross, end_frame=cross + 5, made=made,
                rim=rim, cross_offset=0.1 if made else 2.0)


def test_classifier_overrules_only_confident_makes(tmp_path):
    shots = [_shot(1, True, 20), _shot(2, True, 50), _shot(3, False, 80), _shot(4, True, 110)]
    clf = FakeClassifier([0.95, 0.30, 0.99, 0.81])
    apply_classifier(_video(tmp_path), shots, FPS, 0.8, clf)
    assert [s.made for s in shots] == [False, True, False, False]  # 1 and 4 flipped; 3 was already a miss
    assert [s.make_prob for s in shots] == [0.05, 0.7, 0.01, 0.19]
    assert len(clf.seen) == 4 and clf.seen[0].shape == (IMG_SIZE, IMG_SIZE, 3)


def test_classifier_never_turns_a_miss_into_a_make(tmp_path):
    shots = [_shot(1, False, 40)]
    apply_classifier(_video(tmp_path), shots, FPS, 0.8, FakeClassifier([0.0]))
    assert shots[0].made is False and shots[0].make_prob == 1.0


def test_no_shots_no_work(tmp_path):
    clf = FakeClassifier([])
    apply_classifier(_video(tmp_path), [], FPS, 0.8, clf)
    assert clf.seen == []
