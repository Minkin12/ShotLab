"""Score tracker results against what really happened.

  python -m tracker.main score results.json truth.txt [--make-tolerance 0.6] [--max-exit-speed 1.95]

results.json  the JSON printed by `process ... --no-db`
truth.txt     your own make/miss list, in order. Any of these work:
                miss make make miss ...            (one line or many)
                0 1 1 0 ...                        (1/0, y/n, o/x also accepted)
                1:13.5 make                        (one shot per line with the time in
                73.5 miss                           the video: exact matching by time)
              Lines starting with # are ignored.

--make-tolerance / --max-exit-speed re-score the same results with different
settings (from rim_offset / exit_speed_mps), so you can try values without
re-processing the video.

--classifier-threshold / --rescue also replay the make/miss classifier from the saved
make_prob. Replaying starts again from the geometry, so --make-tolerance and
--max-exit-speed default to 0.6 and 3.0 there; pass them if your .env differs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

MAKE_WORDS = {"make", "made", "m", "1", "y", "yes", "o", "hit", "in"}
MISS_WORDS = {"miss", "missed", "0", "n", "no", "x", "out"}


@dataclass
class TruthShot:
    made: bool
    t: float | None = None  # seconds into the video, if given


@dataclass
class Pair:
    truth: TruthShot | None  # None = extra detection
    row: dict | None  # None = shot not detected


@dataclass
class Report:
    pairs: list[Pair]
    aligned_by: str
    rows: list[dict]
    truth: list[TruthShot]
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _label(tok: str) -> bool | None:
    tok = tok.strip().lower().strip(",;.")
    if tok in MAKE_WORDS:
        return True
    if tok in MISS_WORDS:
        return False
    return None


def _time(tok: str) -> float | None:
    m = re.fullmatch(r"(?:(\d+):)?(\d+(?:\.\d+)?)s?", tok.strip())
    if not m:
        return None
    return (int(m.group(1)) * 60 if m.group(1) else 0) + float(m.group(2))


def parse_truth(text: str) -> list[TruthShot]:
    shots: list[TruthShot] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        toks = re.split(r"[\s,;]+", line)
        # "time label" on its own line -> timed shot ("1:13.5 make", "73.5 miss", "73 make").
        # A bare 0/1 first token is read as a label, so write times as 1:05 or 65.0 there.
        if len(toks) == 2:
            t, lab = _time(toks[0]), _label(toks[1])
            if t is not None and lab is not None and (_label(toks[0]) is None or ":" in toks[0] or "." in toks[0]):
                shots.append(TruthShot(lab, t))
                continue
        for tok in toks:
            lab = _label(tok)
            if lab is None:
                raise ValueError(f"don't understand {tok!r} in truth file (use make/miss)")
            shots.append(TruthShot(lab))
    return shots


def load_rows(path: Path) -> list[dict]:
    text = path.read_text()
    start = text.find("[")  # tolerate log lines before the JSON
    return json.loads(text[start:] if start > 0 else text)


def rescore(
    rows: list[dict],
    make_tolerance: float | None,
    max_exit_speed: float | None,
    classifier_threshold: float | None = None,
    rescue: float | None = None,
    rescue_offset: float = 0.9,
) -> list[dict]:
    """Recompute 'made' with other settings, from the saved rim_offset / exit_speed_mps / make_prob.

    With classifier_threshold or rescue the classifier is replayed after the geometry and
    exit-speed rules, exactly as processing does it.
    """
    replay = classifier_threshold is not None or rescue is not None
    if make_tolerance is None and max_exit_speed is None and not replay:
        return rows
    if replay:
        from .classify import verdict

        if make_tolerance is None:
            raise ValueError("replaying the classifier needs make_tolerance")
        if classifier_threshold is None:
            classifier_threshold = 1.0  # rescue only: never overrule
    out = []
    for r in rows:
        r = dict(r)
        made = r["made"]
        if make_tolerance is not None and r.get("rim_offset") is not None:
            made = r["rim_offset"] <= make_tolerance
        if max_exit_speed is not None and made and (r.get("exit_speed_mps") or 0) > max_exit_speed:
            made = False
        if replay:
            # rescue can undo the exit-speed rule too, as in processing
            made, _ = verdict(made, r.get("rim_offset"), r.get("make_prob"), classifier_threshold,
                              make_tolerance, rescue, rescue_offset)
        r["made"] = made
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def _align_by_time(truth: list[TruthShot], rows: list[dict], window: float) -> list[Pair]:
    used: set[int] = set()
    pairs: list[Pair] = []
    for ts in truth:
        best, best_d = None, window
        for i, r in enumerate(rows):
            d = abs((r.get("video_offset_s") or 0) - ts.t)
            if i not in used and d <= best_d:
                best, best_d = i, d
        if best is None:
            pairs.append(Pair(ts, None))
        else:
            used.add(best)
            pairs.append(Pair(ts, rows[best]))
    pairs += [Pair(None, r) for i, r in enumerate(rows) if i not in used]
    pairs.sort(key=lambda p: p.row["video_offset_s"] if p.row else (p.truth.t or 0))
    return pairs


def _align_by_order(truth: list[TruthShot], rows: list[dict]) -> list[Pair]:
    """Global alignment of the two make/miss sequences (Needleman-Wunsch).

    Pairing any two shots scores +1, pairing with the same result +1 more, a
    gap (missed or extra shot) -1. With equal counts this is a straight 1:1.
    """
    n, m = len(truth), len(rows)
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = -i
    for j in range(1, m + 1):
        score[0][j] = -j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            pair = 1 + (1 if truth[i - 1].made == bool(rows[j - 1]["made"]) else 0)
            score[i][j] = max(score[i - 1][j - 1] + pair, score[i - 1][j] - 1, score[i][j - 1] - 1)
    pairs: list[Pair] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            pair = 1 + (1 if truth[i - 1].made == bool(rows[j - 1]["made"]) else 0)
            if score[i][j] == score[i - 1][j - 1] + pair:
                pairs.append(Pair(truth[i - 1], rows[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and (j == 0 or score[i][j] == score[i - 1][j] - 1):
            pairs.append(Pair(truth[i - 1], None))
            i -= 1
        else:
            pairs.append(Pair(None, rows[j - 1]))
            j -= 1
    return pairs[::-1]


def score(rows: list[dict], truth: list[TruthShot], window: float = 2.5) -> Report:
    timed = [t for t in truth if t.t is not None]
    if truth and len(timed) == len(truth):
        return Report(_align_by_time(truth, rows, window), "time", rows, truth)
    if timed:
        raise ValueError("either give a time for every shot in the truth file, or for none")
    rep = Report(_align_by_order(truth, rows), "order", rows, truth)
    if len(truth) != len(rows):
        rep.notes.append(
            f"{len(truth)} actual vs {len(rows)} detected shots: lined up by order, which is a "
            "best guess - add a time to each line of the truth file for exact matching"
        )
    return rep


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.0f}%" if b else "-"


def _num(r: dict, k: str) -> str:
    v = r.get(k)
    return "-" if v is None else f"{v:g}"


def format_report(rep: Report, title: str = "") -> str:
    paired = [p for p in rep.pairs if p.truth and p.row]
    missed = [p for p in rep.pairs if p.truth and not p.row]
    extra = [p for p in rep.pairs if p.row and not p.truth]
    correct = [p for p in paired if p.truth.made == bool(p.row["made"])]
    false_makes = [p for p in paired if p.row["made"] and not p.truth.made]
    false_misses = [p for p in paired if not p.row["made"] and p.truth.made]
    n_true, n_rows = len(rep.truth), len(rep.rows)
    true_makes = sum(t.made for t in rep.truth)
    app_makes = sum(bool(r["made"]) for r in rep.rows)
    arc = sum(r.get("launch_angle_deg") is not None for r in rep.rows)
    form = sum(r.get("release_elbow_angle") is not None for r in rep.rows)

    def ids(ps: list[Pair]) -> str:
        return ", ".join(f"#{p.row['shot_number']}" for p in ps) if ps else ""

    out = []
    if title:
        out.append(title)
    out += [
        f"Shots:      {n_true} actual, {n_rows} detected "
        f"({len(missed)} missed, {len(extra)} extra)   [matched by {rep.aligned_by}]",
        f"Make/miss:  {len(correct)}/{len(paired)} correct ({_pct(len(correct), len(paired))})",
        f"  actual    {true_makes}/{n_true} makes ({_pct(true_makes, n_true)})",
        f"  app       {app_makes}/{n_rows} makes ({_pct(app_makes, n_rows)})",
        f"  false makes:  {len(false_makes):>3}  {ids(false_makes)}",
        f"  false misses: {len(false_misses):>3}  {ids(false_misses)}",
        f"Arc measured: {arc}/{n_rows} ({_pct(arc, n_rows)})    Form measured: {form}/{n_rows} ({_pct(form, n_rows)})",
    ]

    # exit speed: can it separate real makes from false makes?
    real = [p.row["exit_speed_mps"] for p in paired
            if p.row["made"] and p.truth.made and p.row.get("exit_speed_mps") is not None]
    fake = [p.row["exit_speed_mps"] for p in false_makes if p.row.get("exit_speed_mps") is not None]
    if real and fake:
        hi, lo = max(real), min(fake)
        verdict = (f"separable: a cutoff of ~{(hi + lo) / 2:.2f} m/s would fix all {len(fake)}"
                   if hi < lo else f"overlap: {sum(f <= hi for f in fake)} false make(s) are slower than the fastest real make")
        out.append(f"Exit speed: real makes up to {hi:.2f} m/s, false makes from {lo:.2f} m/s -> {verdict}")

    wrong = false_makes + false_misses
    if wrong or missed or extra:
        out.append("Problems:")
        for p in sorted(wrong, key=lambda p: p.row["shot_number"]):
            r = p.row
            exit_ = f"  exit {r['exit_speed_mps']:g} m/s" if r.get("exit_speed_mps") is not None else ""
            out.append(
                f"  #{r['shot_number']:<3} {r.get('video_offset_s', 0):>7.1f}s  app {'MAKE' if r['made'] else 'miss'}, "
                f"actual {'make' if p.truth.made else 'miss'}   rim_offset {_num(r, 'rim_offset')}{exit_}"
            )
        for p in missed:
            when = f" at {p.truth.t:.1f}s" if p.truth.t is not None else ""
            out.append(f"  not detected: a {'make' if p.truth.made else 'miss'}{when}")
        for p in extra:
            r = p.row
            out.append(f"  #{r['shot_number']:<3} {r.get('video_offset_s', 0):>7.1f}s  extra detection (no real shot)")
    out += [f"Note: {n}" for n in rep.notes]
    return "\n".join(out)
