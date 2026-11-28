-- Runs automatically the first time the Postgres container starts.

CREATE TABLE sessions (
    id           SERIAL PRIMARY KEY,
    video_file   TEXT NOT NULL UNIQUE,
    recorded_at  TIMESTAMPTZ NOT NULL,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    fps          REAL,
    frame_count  INTEGER,
    duration_s   REAL,
    notes        TEXT
);

CREATE TABLE shots (
    id                    SERIAL PRIMARY KEY,
    session_id            INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    shot_number           INTEGER NOT NULL,
    shot_time             TIMESTAMPTZ NOT NULL,
    video_offset_s        REAL,
    made                  BOOLEAN NOT NULL,
    rim_offset            REAL,      -- 0 = dead centre, 1 = at the rim edge (2D)
    exit_speed_mps        REAL,      -- fall speed just below the rim; fast "makes" are suspect
    make_prob             REAL,      -- make/miss classifier's probability of a make

    -- ball flight
    launch_angle_deg      REAL,
    entry_angle_deg       REAL,
    release_speed_mps     REAL,
    apex_above_rim_m      REAL,

    -- form (pose)
    shooting_hand         TEXT,
    release_elbow_angle   REAL,      -- 180 = fully extended
    set_point_elbow_angle REAL,
    min_knee_angle        REAL,      -- lower = deeper dip
    dip_to_release_ms     REAL,
    release_height_ratio  REAL,      -- wrist height / head height at release
    torso_lean_deg        REAL,      -- + toward rim, - fading away
    jump_ratio            REAL,
    form_confidence       REAL,

    clip_path             TEXT,
    tags                  TEXT[],    -- e.g. {free-throw, corner-3}; set by hand
    truth_made            BOOLEAN,   -- what really happened, from your truth file (tracker eval)
    UNIQUE (session_id, shot_number)
);

CREATE INDEX shots_time_idx ON shots (shot_time);

-- Accuracy of each reviewed session, written by `tracker eval`
CREATE TABLE session_scores (
    session_id   INTEGER PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    scored_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    settings     JSONB,
    actual       INTEGER, detected INTEGER, missed INTEGER, extra INTEGER,
    paired       INTEGER, correct INTEGER, false_makes INTEGER, false_misses INTEGER,
    actual_makes INTEGER, arc_measured INTEGER,
    trained_on   BOOLEAN   -- the make/miss classifier was trained on this session
);

-- One row per session
CREATE VIEW session_summary AS
SELECT s.id                                         AS session_id,
       s.recorded_at,
       s.video_file,
       count(sh.id)                                 AS attempts,
       count(sh.id) FILTER (WHERE sh.made)          AS makes,
       round(100.0 * count(sh.id) FILTER (WHERE sh.made) / nullif(count(sh.id), 0), 1) AS fg_pct,
       round(avg(sh.launch_angle_deg)::numeric, 1)  AS avg_launch_angle,
       round(stddev(sh.launch_angle_deg)::numeric, 1) AS sd_launch_angle,
       round(avg(sh.release_elbow_angle)::numeric, 1) AS avg_release_elbow,
       round(stddev(sh.release_elbow_angle)::numeric, 1) AS sd_release_elbow,
       round(avg(sh.min_knee_angle)::numeric, 1)    AS avg_knee_angle,
       round(avg(sh.dip_to_release_ms)::numeric)    AS avg_dip_to_release_ms
FROM sessions s
LEFT JOIN shots sh ON sh.session_id = s.id
GROUP BY s.id;

-- Average form on makes vs misses (only shots where the pose was trustworthy)
CREATE VIEW form_make_vs_miss AS
SELECT CASE WHEN made THEN 'Makes' ELSE 'Misses' END AS result,
       count(*)                                       AS shots,
       round(avg(launch_angle_deg)::numeric, 1)       AS launch_angle,
       round(avg(entry_angle_deg)::numeric, 1)        AS entry_angle,
       round(avg(release_speed_mps)::numeric, 2)      AS release_speed,
       round(avg(release_elbow_angle)::numeric, 1)    AS release_elbow,
       round(avg(set_point_elbow_angle)::numeric, 1)  AS set_point_elbow,
       round(avg(min_knee_angle)::numeric, 1)         AS knee_bend,
       round(avg(dip_to_release_ms)::numeric)         AS dip_to_release_ms,
       round(avg(release_height_ratio)::numeric, 2)   AS release_height,
       round(avg(torso_lean_deg)::numeric, 1)         AS torso_lean,
       round(avg(jump_ratio)::numeric, 3)             AS jump
FROM shots
WHERE form_confidence IS NULL OR form_confidence >= 0.5
GROUP BY made;
