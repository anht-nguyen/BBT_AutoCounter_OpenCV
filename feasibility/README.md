# Feasibility Prototype: BBT Crossing Detection

This folder contains the current OpenCV-based feasibility prototype for detecting Box and Block Test (BBT) transfer-like crossing events from fixed-camera video.

The codebase is organized as a small exploratory pipeline:

1. Define the BBT environment from a reference image.
2. Apply that environment overlay to raw videos.
3. Detect and clean motion blobs near the partition.
4. Score crossing events with a rule-based counter.

This is still a feasibility prototype, not a clinical scoring system.

## Current Folder Layout

```text
feasibility/
+-- README.md
+-- data/
|   +-- images/
|   |   +-- BBT-setup-image.jpg
|   |   `-- annotations/
|   |       +-- BBT_environment.json
|   |       +-- BBT_environment_preview.jpg
|   |       `-- masks/
|   |           +-- crossing_zone_mask.png
|   |           +-- start_side_mask.png
|   |           `-- target_side_mask.png
|   `-- videos/
|       +-- raw/
|       |   +-- BBT-all_incorrect.mp4
|       |   +-- BBT-ground_truth.mp4
|       |   `-- BBT-mixed.mp4
|       `-- annotated/
|           +-- BBT-all_incorrect_environment_overlay.mp4
|           +-- BBT-all_incorrect_motion_detection.mp4
|           +-- BBT-ground_truth_environment_overlay.mp4
|           `-- BBT-ground_truth_motion_detection.mp4
+-- exploratory/
|   +-- 00_define_BBT_environment_image.py
|   +-- 01_define_BBT_environment_video.py
|   +-- 02_detect_motion.py
|   +-- 03_clean_motion_mask.py
|   +-- 04_detect_crossing_event.py
|   `-- all_exploratory_steps.py
+-- src/
|   +-- crossing_event_detector.py
|   +-- define_BBT_environment.py
|   +-- detect_hand_motion_blob.py
|   `-- motion_mask_cleaning.py
`-- utility/
    `-- export_first_frame.py
```

## What The Prototype Currently Does

- Lets you click a box polygon and partition line on a reference image.
- Saves that environment as JSON plus preview masks.
- Reuses the saved environment to overlay start side, target side, and crossing zone on video.
- Uses `cv2.createBackgroundSubtractorMOG2(...)` to generate foreground motion masks.
- Cleans masks with Gaussian blur, thresholding, opening, and closing.
- Extracts the largest motion blob inside the crossing zone.
- Scores transfer-like events with a configurable `CrossingCounter`.
- Supports two crossing methods: `center` and `leading_edge`.
- Can display live debug views and save annotated videos.

## Dependencies

The prototype uses Python with OpenCV and NumPy.

```bash
pip install opencv-python numpy
```

Note: the scripts use OpenCV GUI windows (`cv2.imshow`, mouse callbacks, keyboard controls), so they are meant to be run in a local desktop session rather than a headless environment.

## Main Workflow

Run commands from the repository root.

### 1. Define the Environment On A Reference Image

`exploratory/00_define_BBT_environment_image.py` opens `data/images/BBT-setup-image.jpg` and lets you annotate:

- the 4-point outer box polygon
- the 2-point partition line

It saves:

- `data/images/annotations/BBT_environment.json`
- `data/images/annotations/BBT_environment_preview.jpg`
- `data/images/annotations/masks/start_side_mask.png`
- `data/images/annotations/masks/target_side_mask.png`
- `data/images/annotations/masks/crossing_zone_mask.png`

Command:

```bash
python feasibility/exploratory/00_define_BBT_environment_image.py
```

Current defaults:

- `CROSSING_ZONE_WIDTH = 90`
- `START_SIDE = "left"`

### 2. Preview The Environment On Video

`exploratory/01_define_BBT_environment_video.py` loads the saved environment and overlays it on a raw video.

Command:

```bash
python feasibility/exploratory/01_define_BBT_environment_video.py --video feasibility/data/videos/raw/BBT-ground_truth.mp4
```

By default it writes an annotated preview video to:

```text
feasibility/data/videos/annotated/<video_stem>_environment_overlay.mp4
```

### 3. Detect Motion In The Crossing Zone

`exploratory/02_detect_motion.py` loads the environment JSON, builds a crossing-zone mask, detects motion blobs, and shows:

- an annotated video view
- a debug panel with original frame, motion mask, and crossing-zone mask

Command:

```bash
python feasibility/exploratory/02_detect_motion.py --video feasibility/data/videos/raw/BBT-ground_truth.mp4
```

Default detector settings:

- `history = 500`
- `var_threshold = 50`
- `detect_shadows = False`
- `blur_kernel_size = (5, 5)`
- `threshold_value = 200`
- `morphology_kernel_size = (5, 5)`
- `min_area = 500`

By default it writes:

```text
feasibility/data/videos/annotated/<video_stem>_motion_detection.mp4
```

### 4. Inspect Motion Mask Cleaning

`exploratory/03_clean_motion_mask.py` is a focused debugging view for the mask-cleaning step. It displays:

- original frame
- raw foreground mask
- cleaned mask
- contour debug view

Command:

```bash
python feasibility/exploratory/03_clean_motion_mask.py --video feasibility/data/videos/raw/BBT-ground_truth.mp4
```

Current display defaults in this script:

- resize to `1280 x 720`
- blur kernel `5`
- threshold `200`
- morphology kernel `5`
- opening iterations `1`
- closing iterations `1`
- `min_area = 500`

### 5. Inspect Crossing Event Detection

`exploratory/04_detect_crossing_event.py` runs the current scoring logic and shows a four-panel debug view:

- original frame
- cleaned motion mask
- contour debug
- scoring/count overlay

Command:

```bash
python feasibility/exploratory/04_detect_crossing_event.py --video feasibility/data/videos/raw/BBT-ground_truth.mp4
```

Current scoring defaults in this script:

- `PARTITION_X = 640`
- `DIRECTION = "left_to_right"`
- `CROSSING_ZONE_WIDTH = 100`
- `DEAD_ZONE_WIDTH = 20`
- `MIN_AREA = 500`
- `COOLDOWN_FRAMES = 20`
- `CROSSING_METHOD = "leading_edge"`
- `LEADING_EDGE_MARGIN = 10`

### 6. Run Steps 01 To 04 Across Selected Videos

`exploratory/all_exploratory_steps.py` prompts you to choose one or more raw videos from `data/videos/raw/`, then runs:

1. `01_define_BBT_environment_video.py`
2. `02_detect_motion.py`
3. `03_clean_motion_mask.py`
4. `04_detect_crossing_event.py`

Command:

```bash
python feasibility/exploratory/all_exploratory_steps.py
```

## Reusable Modules In `src`

### `define_BBT_environment.py`

Core environment-annotation utilities:

- `BBTEnvironment`
- `annotate_environment(...)`
- `build_side_polygons(...)`
- `build_crossing_zone_polygon(...)`
- `save_environment(...)`
- `save_environment_preview(...)`
- `save_region_masks(...)`

The saved environment JSON is the key input for later video-based steps.

### `detect_hand_motion_blob.py`

Motion-detection utilities built around the saved environment:

- `MotionEnvironment`
- `MotionDetectorConfig`
- `HandMotionBlobDetector`
- `annotate_motion_frame(...)`
- `build_debug_panel(...)`
- `load_environment(...)`

The detector restricts analysis to the annotated crossing zone and returns the largest valid foreground blob.

### `motion_mask_cleaning.py`

Standalone mask-processing utilities:

- `clean_motion_mask(...)`
- `filter_contours_by_area(...)`
- `draw_contour_debug(...)`
- `resize_frame(...)`

This module is used directly by the exploratory scripts and by the crossing detector.

It also has its own CLI:

```bash
python feasibility/src/motion_mask_cleaning.py --video feasibility/data/videos/raw/BBT-ground_truth.mp4
```

### `crossing_event_detector.py`

Rule-based crossing counter and CLI entry point.

The main class is `CrossingCounter`, which tracks:

- count state
- previous side
- dead-zone behavior
- cooldown frames
- arming/re-arming logic
- either center-based or leading-edge crossing detection

It also exposes a CLI:

```bash
python feasibility/src/crossing_event_detector.py \
  --video feasibility/data/videos/raw/BBT-ground_truth.mp4 \
  --partition-x 640 \
  --direction left_to_right \
  --crossing-zone-width 100 \
  --dead-zone-width 20 \
  --min-area 500 \
  --cooldown-frames 20 \
  --crossing-method leading_edge \
  --leading-edge-margin 10 \
  --display \
  --save-output feasibility/data/videos/annotated/BBT-ground_truth_crossing_debug.mp4
```

Supported CLI arguments:

- `--video`
- `--partition-x`
- `--direction`
- `--crossing-zone-width`
- `--dead-zone-width`
- `--min-area`
- `--cooldown-frames`
- `--crossing-method`
- `--leading-edge-margin`
- `--resize-width`
- `--resize-height`
- `--display`
- `--save-output`

## Utility Script

`utility/export_first_frame.py` exports the first frame from a chosen video. You can pass a video path directly or let it open a file picker.

Examples:

```bash
python feasibility/utility/export_first_frame.py feasibility/data/videos/raw/BBT-ground_truth.mp4
```

```bash
python feasibility/utility/export_first_frame.py feasibility/data/videos/raw/BBT-ground_truth.mp4 -o feasibility/data/images/BBT-setup-image.jpg
```

## Current Data Assumptions

The prototype currently assumes:

- fixed-camera video
- a visible partition
- a manually defined environment from a reference image
- one dominant moving blob near the crossing zone
- simple rule-based scoring rather than object-level block tracking

The current implementation is best understood as a controlled proof-of-concept for motion-based event counting.

## Existing Example Assets

Current raw videos:

- `data/videos/raw/BBT-ground_truth.mp4`
- `data/videos/raw/BBT-all_incorrect.mp4`
- `data/videos/raw/BBT-mixed.mp4`

Current saved image/environment assets:

- `data/images/BBT-setup-image.jpg`
- `data/images/annotations/BBT_environment.json`
- `data/images/annotations/BBT_environment_preview.jpg`

## Practical Notes

- Press `q` to quit most preview windows.
- Press space in the video preview scripts to pause on the current frame.
- The exploratory scripts use hard-coded defaults intended for inspection and iteration, not for batch evaluation.
- `04_detect_crossing_event.py` currently uses a fixed `PARTITION_X = 640`; it does not yet derive that value automatically from `BBT_environment.json`.

## Current Status

The current repo state is strongest as an interactive exploratory toolkit for:

- annotating the environment
- validating the crossing zone
- inspecting motion masks
- tuning blob filtering
- comparing center-based vs leading-edge crossing logic

It is not yet a full evaluation pipeline with ground-truth tables, summary metrics, or automated experiment reporting.
