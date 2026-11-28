"""`tracker eval`: score every reviewed session in one go.

Upload right after a session; review it whenever you like by writing
VIDEO_ROOT/debug/<video name>_truth.txt (same formats as `score`). Then:

  python -m tracker.main eval                  # table of every reviewed session
  python -m tracker.main eval --details        # plus each session's problem shots
  python -m tracker.main eval --export-crops   # also refresh classifier training images

Results come from the database (what the watcher saved), so nothing is re-processed.
They are re-scored with the *current* settings (.env, or the --flags) the same way
processing does it, so you can try a setting on every session at once before
changing it. Each run saves the scores (table session_scores) and every shot's
real result (shots.truth_made) for Grafana; the app's own make/miss is not changed.

Sessions the make/miss classifier was trained on are marked: its make_prob is
optimistic there, so judge the classifier on the other sessions.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .config import Config
from .score import Report, format_report, load_rows, parse_truth, report_counts, rescore, score

log = logging.getLogger(__name__)

TRUTH_SUFFIX = "_truth.txt"
DEFAULT_DATASET = Path("/models/datasets/makecls_all")


@dataclass
class Settings:
    make_tolerance: float
    max_exit_speed: float | None
    classifier_threshold: float | None
    rescue: float | None
    rescue_offset: float

    @classmethod
    def from_config(cls, cfg: Config) -> "Settings":
        on = cfg.make_classifier is not None
        return cls(
            make_tolerance=cfg.make_tolerance,
            max_exit_speed=cfg.max_make_exit_speed,
            classifier_threshold=cfg.make_classifier_threshold if on else None,
            rescue=cfg.make_classifier_rescue if on else None,
            rescue_offset=cfg.make_classifier_rescue_offset,
        )

    def describe(self) -> str:
        f = lambda v: "off" if v is None else f"{v:g}"  # noqa: E731
        return (f"make_tolerance={f(self.make_tolerance)}  max_exit_speed={f(self.max_exit_speed)}  "
                f"classifier_threshold={f(self.classifier_threshold)}  rescue={f(self.rescue)}"
                + (f" (up to rim_offset {self.rescue_offset:g})" if self.rescue is not None else ""))


@dataclass
class Session:
    video_file: str
    rows: list[dict]
    session_id: int | None = None  # None = read from a JSON file, not the database
    recorded_at: datetime | None = None

    @property
    def stem(self) -> str:
        return Path(self.video_file).stem


@dataclass
class Result:
    session: Session
    truth_path: Path
    report: Report
    counts: dict
    trained_on: bool
    crops: str = ""  # what --export-crops did
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def truth_files(debug_dir: Path) -> dict[str, Path]:
    """{video stem: truth file} for every <stem>_truth.txt in the debug folder."""
    return {p.name[: -len(TRUTH_SUFFIX)]: p for p in sorted(debug_dir.glob(f"*{TRUTH_SUFFIX}"))}


def db_sessions(dsn: str) -> list[Session]:
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        sessions = conn.execute("SELECT id, video_file, recorded_at FROM sessions ORDER BY recorded_at").fetchall()
        out = []
        for s in sessions:
            rows = conn.execute(
                "SELECT * FROM shots WHERE session_id = %s ORDER BY shot_number", (s["id"],)
            ).fetchall()
            out.append(Session(s["video_file"], rows, s["id"], s["recorded_at"]))
        return out


def load_sessions(cfg: Config, use_db: bool = True) -> list[Session]:
    """Sessions from the database, plus <stem>.json results (`process --no-db`) for any
    reviewed video that isn't in the database."""
    sessions: list[Session] = []
    if use_db:
        try:
            sessions = db_sessions(cfg.database_url)
        except Exception as e:  # noqa: BLE001
            log.warning("database not reachable (%s) - using JSON results only", e)
    have = {s.stem for s in sessions}
    for stem in truth_files(cfg.debug):
        js = cfg.debug / f"{stem}.json"
        if stem not in have and js.exists():
            sessions.append(Session(f"{stem}.json", load_rows(js)))
    return sessions


def trained_on(stem: str, dataset: Path, model: str | None) -> bool:
    """True if the session's images are in the training set and the classifier file is
    newer than them (it was trained after they were added)."""
    crops = _crop_files(stem, dataset / "train")
    if not crops:
        return False
    m = Path(model) if model else None
    if m is None or not m.exists():
        return True
    return max(p.stat().st_mtime for p in crops) < m.stat().st_mtime


def _crop_files(stem: str, split_dir: Path) -> list[Path]:
    pat = re.compile(rf"^{re.escape(stem)}_\d{{3}}\.jpg$")  # exact stem: form_shooting != form_shooting_2026...
    return [p for d in (split_dir / "make", split_dir / "miss") if d.is_dir() for p in d.iterdir() if pat.match(p.name)]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def evaluate(sessions: list[Session], truths: dict[str, Path], st: Settings, dataset: Path,
             model: str | None = None, window: float = 2.5) -> tuple[list[Result], list[Session]]:
    """Returns (results for reviewed sessions, sessions with no truth file yet)."""
    results, unreviewed = [], []
    for s in sessions:
        tp = truths.get(s.stem)
        if tp is None:
            unreviewed.append(s)
            continue
        rows = rescore(s.rows, st.make_tolerance, st.max_exit_speed,
                       classifier_threshold=st.classifier_threshold, rescue=st.rescue,
                       rescue_offset=st.rescue_offset)
        rep = score(rows, parse_truth(tp.read_text()), window)
        results.append(Result(s, tp, rep, report_counts(rep), trained_on(s.stem, dataset, model)))
    return results, unreviewed


def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:5.1f}%" if b else "    -"


def format_table(results: list[Result], unreviewed: list[Session], st: Settings, details: bool = False) -> str:
    w = max([len(r.session.video_file) for r in results] + [24])
    head = f"{'session':<{w}}  {'shots':>9}  {'make/miss':>16}  {'FM':>3} {'FN':>3}  {'arc':>9}  note"
    out = [f"Settings: {st.describe()}", "", head, "-" * len(head)]

    def line(name: str, c: dict, note: str = "") -> str:
        shots = f"{c['detected']}/{c['actual']}"
        mm = f"{c['correct']}/{c['paired']} {_pct(c['correct'], c['paired'])}"
        arc = f"{c['arc_measured']}/{c['detected']}"
        return f"{name:<{w}}  {shots:>9}  {mm:>16}  {c['false_makes']:>3} {c['false_misses']:>3}  {arc:>9}  {note}"

    def total(rs: list[Result]) -> dict:
        return {k: sum(r.counts[k] for r in rs) for k in results[0].counts} if rs else {}

    for r in results:
        note = ", ".join(filter(None, ["classifier trained on it" if r.trained_on else "",
                                       r.crops] + r.notes))
        if r.counts["missed"] or r.counts["extra"]:
            note = ", ".join(filter(None, [f"{r.counts['missed']} missed, {r.counts['extra']} extra", note]))
        out.append(line(r.session.video_file, r.counts, note))
    if results:
        out.append("-" * len(head))
        out.append(line("ALL", total(results)))
        unseen = [r for r in results if not r.trained_on]
        if len(unseen) not in (0, len(results)):
            out.append(line("not trained on (honest)", total(unseen)))
    out.append("FM = false makes, FN = false misses. shots = detected/actual.")
    if unreviewed:
        out.append("")
        out.append(f"No truth file yet ({len(unreviewed)}): " + ", ".join(s.video_file for s in unreviewed))
        out.append(f"  add <video name>{TRUTH_SUFFIX} to the debug folder to include them")
    if details:
        for r in results:
            if r.counts["correct"] < r.counts["paired"] or r.counts["missed"] or r.counts["extra"]:
                out += ["", format_report(r.report, f"== {r.session.video_file}")]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Saving and training images
# --------------------------------------------------------------------------- #
SCORE_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS session_scores (
        session_id   INTEGER PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
        scored_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        settings     JSONB,
        actual       INTEGER, detected INTEGER, missed INTEGER, extra INTEGER,
        paired       INTEGER, correct INTEGER, false_makes INTEGER, false_misses INTEGER,
        actual_makes INTEGER, arc_measured INTEGER,
        trained_on   BOOLEAN
    )""",
    "ALTER TABLE shots ADD COLUMN IF NOT EXISTS truth_made BOOLEAN",
]


def save_scores(dsn: str, results: list[Result], st: Settings) -> int:
    import psycopg

    saved = 0
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        for sql in SCORE_SCHEMA:
            cur.execute(sql)
        for r in results:
            sid = r.session.session_id
            if sid is None:
                continue
            c = r.counts
            cur.execute(
                """INSERT INTO session_scores (session_id, settings, actual, detected, missed, extra, paired,
                       correct, false_makes, false_misses, actual_makes, arc_measured, trained_on)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (session_id) DO UPDATE SET scored_at = now(), settings = EXCLUDED.settings,
                       actual = EXCLUDED.actual, detected = EXCLUDED.detected, missed = EXCLUDED.missed,
                       extra = EXCLUDED.extra, paired = EXCLUDED.paired, correct = EXCLUDED.correct,
                       false_makes = EXCLUDED.false_makes, false_misses = EXCLUDED.false_misses,
                       actual_makes = EXCLUDED.actual_makes, arc_measured = EXCLUDED.arc_measured,
                       trained_on = EXCLUDED.trained_on""",
                (sid, json.dumps(asdict(st)), c["actual"], c["detected"], c["missed"], c["extra"], c["paired"],
                 c["correct"], c["false_makes"], c["false_misses"], c["actual_makes"], c["arc_measured"],
                 r.trained_on),
            )
            cur.execute("UPDATE shots SET truth_made = NULL WHERE session_id = %s", (sid,))
            cur.executemany(
                "UPDATE shots SET truth_made = %s WHERE session_id = %s AND shot_number = %s",
                [(p.truth.made, sid, p.row["shot_number"]) for p in r.report.pairs if p.truth and p.row],
            )
            saved += 1
        conn.commit()
    return saved


def refresh_crops(r: Result, cfg: Config, dataset: Path) -> str:
    """(Re-)export a reviewed session's training images when it has none yet, or when
    the truth file changed since they were made. Returns a short note for the table."""
    stem = r.session.stem
    tracks = cfg.debug / f"{stem}_tracks.json"
    video = cfg.processed / r.session.video_file
    if not video.exists() or not tracks.exists():
        return "no crops (needs the video in processed/ and its _tracks.json)"
    existing = _crop_files(stem, dataset / "train")
    if existing and max(p.stat().st_mtime for p in existing) >= r.truth_path.stat().st_mtime:
        return ""  # up to date
    from .crops import export_crops

    for p in existing:
        p.unlink()
    c = export_crops(video, tracks, r.truth_path, dataset, "train")
    r.trained_on = trained_on(stem, dataset, cfg.make_classifier)  # new images: not trained on yet
    return f"{'re-' if existing else ''}exported {c['make']} make/{c['miss']} miss images"


def run(cfg: Config, st: Settings, *, details: bool = False, save: bool = True, export: bool = False,
        dataset: Path = DEFAULT_DATASET, only: list[str] | None = None, use_db: bool = True) -> str:
    sessions = load_sessions(cfg, use_db)
    if only:
        sessions = [s for s in sessions if s.stem in only or s.video_file in only]
    results, unreviewed = evaluate(sessions, truth_files(cfg.debug), st, dataset, cfg.make_classifier)
    exported = 0
    if export:
        for r in results:
            r.crops = refresh_crops(r, cfg, dataset)
            exported += r.crops.startswith(("exported", "re-exported"))
    text = format_table(results, unreviewed, st, details)
    if save and use_db and results:
        try:
            n = save_scores(cfg.database_url, results, st)
            text += f"\nSaved scores for {n} session(s) to the database (Grafana: Accuracy panel)."
        except Exception as e:  # noqa: BLE001
            text += f"\nCould not save scores: {e}"
    if exported:
        text += (f"\n{exported} session(s) have new training images in {dataset}/train - retrain the "
                 "classifier to use them (README section 7b).")
    return text
