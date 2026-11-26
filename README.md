# Shot Tracker

A self-hosted basketball shot tracker. You put a video of a shooting session into a folder on your NAS. The worker then:

- detects every shot and whether it went in (the "make/miss" result)
- measures the ball's flight: launch angle, entry angle, release speed and how high the arc goes
- analyzes your shooting form from body pose: elbow extension, set point, knee dip, rhythm, release height, lean and jump
- cuts a short clip of each shot (optional)
- saves everything to Postgres, which feeds a Grafana dashboard you can view over time

All processing happens on your own hardware. Nothing goes to the cloud.

```
phone / GoPro / PoE cam
        │  upload
        ▼
NAS: basketball/incoming/  ──►  tracker (GPU)  ──►  Postgres  ──►  Grafana :3000
                                   │
                                   └──► clips/<session>/003_make.mp4, processed/, debug/
```

---

## 1. Get a ball + rim detection model

The pose model (`yolo11s-pose.pt`) downloads automatically. You need to supply the **ball and rim** model yourself. It goes in `models/ball_rim.pt`.

**Option A: train your own (recommended, about an hour of mostly waiting)**

1. Download a basketball dataset from [Roboflow Universe](https://universe.roboflow.com) (search "basketball hoop"). Pick one with both a ball class and a rim/hoop class, and export it in **YOLO** format.
2. Train it on your GPU:
   ```bash
   pip install ultralytics
   yolo detect train model=yolo11s.pt data=path/to/data.yaml epochs=100 imgsz=960
   cp runs/detect/train/weights/best.pt models/ball_rim.pt
   ```
3. **The biggest accuracy boost:** record one session on *your* court, grab 100–200 frames from it (`ffmpeg -i session.mp4 -vf fps=2 frames/%04d.jpg`), label them in Roboflow or Label Studio, and fine-tune `best.pt` on them. Your lighting, rim and background will then be in the training data.

**Option B: use existing weights.** Several open-source shot detector repos publish a trained `best.pt`. Check each one's license before using it.

Whichever you use, set `BALL_CLASSES` / `RIM_CLASSES` in `.env` to match the model's class names (case doesn't matter). The worker refuses to start if they don't match, and its error message lists the classes the model actually has.

## 2. Camera placement

The camera position matters more than anything else here.

- **Side-on.** Put the camera roughly perpendicular to the line from you to the rim, far enough back that **both you and the rim** are in the frame. Form angles and launch angles are only real angles from this view. At other views they're still consistent from shot to shot, but inflated.
- **60 fps, 1080p** if your camera supports it. 30 fps works, but release detection gets coarser.
- **Tripod, not handheld.** The rim should stay still in the frame.
- Lock the exposure on a phone so the image doesn't pulse when you walk through bright spots.
- Moving the camera between sessions is fine. Moving it *during* a session isn't.

## 3. Run it

Requirements: Docker, plus the NVIDIA Container Toolkit for the GPU (see `docker-compose.yml` for running CPU-only).

```bash
cp .env.example .env        # set POSTGRES_PASSWORD and VIDEO_DIR at minimum
mkdir -p models && cp /path/to/best.pt models/ball_rim.pt
docker compose up -d --build
```

Open Grafana at `http://<server>:3000` (user `admin`, password `GRAFANA_ADMIN_PASSWORD`). The **Shot Tracker** dashboard is the home page.

### Day to day

Copy a session video into `VIDEO_DIR/incoming/`. The worker checks that folder every `POLL_SECONDS`. It waits until the file has finished uploading, processes it, then moves it to `processed/`. If processing fails, the video goes to `failed/` along with a `.log` file.

Re-processing a file (moving it back into `incoming/`) replaces that session's data, so it's safe to tune settings and run it again.

### Test a single video first

```bash
docker compose run --rm tracker process /videos/incoming/test.mp4 --debug --no-db
```

This prints every detected shot as JSON and writes `debug/test_debug.mp4`. That copy has the ball, rim, skeleton, release point and running tally drawn on it. Watch it before trusting the numbers.

## 4. Tuning

| Symptom | Fix |
|---|---|
| Ball missed when it's small or far away | Raise `IMGSZ` (1280), lower `BALL_CONF`, or fine-tune on your court |
| Rim-out shots counted as makes / swishes counted as misses | Adjust `MAKE_TOLERANCE` (default 0.7, where 1.0 = rim edge). Check the `rim_offset` column for borderline shots |
| Passes or rebounds that go above the rim counted as shots | Raise `SHOT_COOLDOWN_S`, or move the camera so the rim is near the edge of the frame |
| Wrong hand detected | Set `SHOOTING_HAND=right` or `left` |
| Form metrics empty | The pose wasn't visible enough. Check `form_confidence` and move the camera closer or side-on |

## 5. What the numbers mean

These are all 2D measurements from the camera's point of view. Compare **your makes with your misses** rather than with textbook numbers. The "Form: makes vs misses" table on the dashboard shows exactly that.

**Ball flight**

| Column | Meaning |
|---|---|
| `launch_angle_deg` | Angle the ball leaves your hand at. Around 45–55° is typical for jump shots |
| `entry_angle_deg` | Angle the ball comes down into the rim at. Steeper gives a bigger target |
| `release_speed_mps` | Ball speed at release. The scale comes from fitting gravity to the arc |
| `apex_above_rim_m` | How high the arc peaks above the rim |
| `rim_offset` | Where the ball crossed the rim plane: 0 = center, 1 = edge |

**Form (pose)**

| Column | Meaning |
|---|---|
| `release_elbow_angle` | Shoulder-elbow-wrist angle at release (180 = arm fully extended) |
| `set_point_elbow_angle` | Tightest elbow angle with the ball above your shoulder before you release |
| `min_knee_angle` | Deepest knee bend in the second before release. Lower means a deeper dip |
| `dip_to_release_ms` | Time from your deepest dip to release. Shows your rhythm |
| `release_height_ratio` | Wrist height at release ÷ head height. Above 1 means you released above your head |
| `torso_lean_deg` | Torso tilt at release: + = toward the rim, − = fading away |
| `jump_ratio` | How far your ankles lifted at release ÷ your height. 0 = set shot |
| `form_confidence` | Pose keypoint confidence. The dashboard ignores form data below 0.5 |

**How release is detected:** the release is the last frame where your shooting wrist is within about one ball-width of the ball before the ball rises above the rim. If no pose is detected, it's estimated from the ball's arc instead: the worker extends the fitted arc backward until the points stop matching it.

## 6. Dashboard

- **Top row:** attempts, makes, FG%, average launch angle, average release elbow angle, number of sessions
- **FG % by session**, and **consistency by session** (the spread of launch angle and elbow angle within each session; lower = more repeatable)
- **Form: makes vs misses:** every metric averaged separately for makes and misses
- **Make % by launch angle / elbow angle / knee bend:** shows your best range for each
- **Fatigue check:** metrics shot by shot. Zoom into one session to see whether your legs or arc drop off as you tire
- **Shots:** every shot, with a link to its clip

Clips are stored under `VIDEO_DIR/clips/`. If you also point Jellyfin at that folder, you get a highlight library.

## 7. Tagging shots

The `shots.tags` column is there for tagging by hand, for example:

```sql
UPDATE shots SET tags = '{free-throw}' WHERE session_id = 12;
```

You can then filter any dashboard panel with `AND 'free-throw' = ANY(tags)`.

## 7b. Make/miss classifier (optional)

From the side, a ball rattling out or dropping just behind the rim can look like a make.
A small image classifier fixes most of these: it looks at the rim just before and after
the ball's final crossing (the net snapping on a make is the strongest clue).

1. Mark sessions by hand (`truth.txt`, same formats as `score`) and process them with `--dump-tracks`.
2. Export training images, one session held out for validation:
   ```bash
   docker compose run --rm --no-deps tracker crops /videos/processed/<video> \
     /videos/debug/<video>_tracks.json /videos/debug/<video>_truth.txt --out /models/datasets/makecls --split train --preview
   ```
3. Train: `yolo classify train model=yolo11n-cls.pt data=/models/datasets/makecls imgsz=224 epochs=50 batch=32`
4. Copy `best.pt` to `models/make_cls.pt` and set `MAKE_CLASSIFIER=/models/make_cls.pt` in `.env`.

Every shot then gets `make_prob`; a make whose p(miss) is above `MAKE_CLASSIFIER_THRESHOLD`
(default 0.8) is scored a miss and logged. Shots already scored a miss are never changed.
Retrain as you mark more sessions.

## 8. Ideas for later

- **Shot chart:** map court lines to pixel positions (a homography) to plot where each shot came from and get percentages by zone.
- **Live mode:** read an RTSP stream from a PoE camera and push makes and misses to a tablet or e-ink display as you shoot.
- **Swish vs. rim:** check whether the ball's path stays clean after the crossing to separate swishes from rim-ins.

## Development

The analysis code (`shots.py`, `form.py`) only needs numpy, so you can test it without a GPU or video:

```bash
cd worker && pip install numpy pytest && PYTHONPATH=. pytest tests
```

The tests generate synthetic shots (a ballistic ball flight and a shooter with a knee dip) and check make/miss, launch angle, release speed, release frame, hand detection and knee bend. They also add realistic detection noise and dropped frames.

Note: `ultralytics` is licensed AGPL-3.0. That's fine for a personal homelab project.
