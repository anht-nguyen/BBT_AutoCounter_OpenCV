# Technical Feasibility Test: Crossing-Line Detection for Box and Block Test Scoring

This subfolder contains a technical feasibility prototype for testing whether a simple computer vision approach can estimate Box and Block Test (BBT) scores from fixed-camera video.

The goal is **not** to build a final clinical-grade scoring system. The goal is to answer a focused feasibility question:

> Can a controlled video setup and a simple crossing-line detection algorithm reliably detect block-transfer events during a Box and Block Test–like task?

This feasibility test supports a larger summer research project on computer vision–assisted BBT scoring.

---

## 1. Project Scope

The Box and Block Test is commonly scored by counting the number of blocks transferred over a partition within a fixed time window, typically 60 seconds. In this prototype, we test whether a computer vision algorithm can detect transfer events by identifying motion crossing a virtual line aligned with the box partition.

This subproject focuses on **Option A: crossing-line detection**.

The core idea is simple:

1. Record a BBT-like trial using a fixed camera.
2. Define a virtual counting line at the partition.
3. Detect motion near the partition.
4. Count one event when the moving hand/block crosses from the start side to the target side.
5. Compare the automated count with manual scoring.

---

## 2. Feasibility Question

The primary feasibility question is:

> Under controlled recording conditions, can a simple OpenCV-based crossing-line algorithm estimate BBT transfer counts with acceptable error compared with manual scoring?

Secondary questions:

- Does the algorithm work under ideal lighting and fixed camera conditions?
- How sensitive is the algorithm to hand occlusion?
- Does speed of movement affect counting accuracy?
- What are the most common causes of false positives and missed counts?
- Is this approach appropriate for a 6-week high school summer research project?

---

## 3. What This Prototype Does

This prototype is designed to:

- Load a recorded BBT-like video.
- Resize and process video frames.
- Define a partition line and crossing zone.
- Use motion detection to identify movement near the partition.
- Track whether the detected motion crosses from one side to the other.
- Count valid crossing events.
- Display the current count on the video.
- Optionally save an annotated output video.
- Compare automated count against manual count.

---

## 4. What This Prototype Does Not Do

This prototype does **not** currently:

- Detect individual blocks with high precision.
- Distinguish perfectly between hand motion and block motion.
- Handle arbitrary camera angles.
- Handle uncontrolled lighting.
- Validate performance in clinical populations.
- Replace human scoring.
- Guarantee perfect scoring during real BBT administration.

The prototype should be treated as a **proof-of-concept feasibility test**.

---

## 5. Recommended Folder Structure

```text
crossing_line_feasibility/
│
├── README.md
├── src/
│   ├── crossing_line_counter.py
│   └── utils.py
│
├── videos/
│   ├── raw/
│   │   ├── pilot_01_slow.mp4
│   │   ├── pilot_02_normal.mp4
│   │   └── pilot_03_fast.mp4
│   │
│   └── annotated/
│       ├── pilot_01_slow_annotated.mp4
│       └── pilot_02_normal_annotated.mp4
│
├── data/
│   ├── manual_counts.csv
│   └── automated_counts.csv
│
├── results/
│   ├── feasibility_summary.csv
│   ├── error_analysis.csv
│   └── figures/
│       ├── manual_vs_automated.png
│       └── absolute_error_by_video.png
│
└── notes/
    ├── recording_protocol.md
    └── failure_cases.md
```

---

## 6. Recording Setup

Use a controlled setup to make the computer vision problem manageable.

Recommended setup:

- Fixed camera on a tripod or stable mount.
- Camera placed above or slightly in front of the box.
- Clear view of both compartments and the partition.
- Consistent lighting.
- High-contrast background.
- Colored blocks if available.
- Box does not move during recording.
- The partition line is clearly visible.
- One trial per video.

Recommended camera view:

```text
Camera
  ↓
Slightly front-top oblique angle

+-----------------------------+
| Start side  |  Target side  |
|             |               |
|             |               |
+-------------|---------------+
              ^
        partition line
```

Avoid:

- Hand blocking the entire partition.
- Moving camera.
- Dark shadows over the box.
- Highly reflective surfaces.
- Background clutter.
- Camera angles where the partition is not visible.

---

## 7. Pilot Video Conditions

Start with short pilot videos before testing full 60-second trials.

Recommended pilot set:

| Video ID | Condition | Expected Transfers | Purpose |
|---|---:|---:|---|
| pilot_01 | Slow movement, ideal lighting | 10 | Debug algorithm |
| pilot_02 | Normal speed, ideal lighting | 20 | Test basic accuracy |
| pilot_03 | Fast movement | 20 | Test missed counts |
| pilot_04 | More hand occlusion | 20 | Test robustness |
| pilot_05 | Full 60-second simulated trial | 40–70 | Estimate real-world feasibility |

For each video, manually record the true number of successful block transfers.

---

## 8. Manual Count File

Create a CSV file at:

```text
data/manual_counts.csv
```

Recommended format:

```csv
video_id,filename,manual_count,condition,notes
pilot_01,pilot_01_slow.mp4,10,slow ideal lighting,debug video
pilot_02,pilot_02_normal.mp4,20,normal ideal lighting,
pilot_03,pilot_03_fast.mp4,20,fast movement,
pilot_04,pilot_04_occlusion.mp4,20,hand occlusion,
pilot_05,pilot_05_full_trial.mp4,55,full 60-second trial,
```

---

## 9. Algorithm Overview

The crossing-line algorithm follows this logic:

1. Load video.
2. Define the partition line.
3. Define a narrow crossing zone around the partition.
4. Use background subtraction to detect moving pixels.
5. Clean the motion mask using thresholding and morphological operations.
6. Find the largest moving contour near the partition.
7. Estimate the contour center.
8. Determine whether the center is on the start side or target side.
9. Count a transfer when movement crosses in the expected direction.
10. Apply a cooldown period to avoid double-counting.

Example event logic:

```text
If previous side = start side
AND current side = target side
AND cooldown period has passed
THEN count = count + 1
```

For right-hand trials, the expected direction may be left-to-right.

For left-hand trials, the expected direction may be right-to-left.

---

## 10. Key Parameters

The following parameters should be adjusted for each camera setup:

| Parameter | Description | Example |
|---|---|---:|
| `partition_x` | x-coordinate of the partition line | 320 |
| `crossing_zone_width` | width of region around partition | 80 |
| `min_area` | minimum moving blob area to consider | 500 |
| `cooldown_frames` | frames to ignore after a count | 20 |
| `direction` | valid transfer direction | left_to_right |
| `resize_width` | standardized video width | 640 |
| `resize_height` | standardized video height | 480 |

Parameter tuning should be documented in the notes or results file.

---

## 11. Minimal Example Script

Below is a minimal feasibility script. It is intentionally simple and should be improved during testing.

```python
import cv2
import numpy as np

video_path = "videos/raw/pilot_01_slow.mp4"

cap = cv2.VideoCapture(video_path)

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=False
)

partition_x = 320
crossing_zone_width = 80
min_area = 500
cooldown_frames = 20

count = 0
last_count_frame = -cooldown_frames
frame_idx = 0
previous_side = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1
    frame = cv2.resize(frame, (640, 480))

    x1 = partition_x - crossing_zone_width // 2
    x2 = partition_x + crossing_zone_width // 2

    fgmask = fgbg.apply(frame)
    fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    zone_mask = fgmask[:, x1:x2]

    contours, _ = cv2.findContours(
        zone_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    largest_area = 0
    blob_center_x = None

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > largest_area and area > min_area:
            largest_area = area
            x, y, w, h = cv2.boundingRect(contour)
            blob_center_x = x1 + x + w // 2

    if blob_center_x is not None:
        current_side = "left" if blob_center_x < partition_x else "right"

        if previous_side == "left" and current_side == "right":
            if frame_idx - last_count_frame > cooldown_frames:
                count += 1
                last_count_frame = frame_idx

        previous_side = current_side

        cv2.circle(frame, (blob_center_x, 240), 8, (0, 255, 0), -1)

    cv2.line(frame, (partition_x, 0), (partition_x, 480), (0, 0, 255), 2)
    cv2.rectangle(frame, (x1, 0), (x2, 480), (255, 0, 0), 2)

    cv2.putText(
        frame,
        f"Count: {count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.imshow("BBT Crossing-Line Counter", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Final automated count: {count}")
```

---

## 12. Suggested Command-Line Interface

A later version of the script can support command-line arguments:

```bash
python src/crossing_line_counter.py \
    --video videos/raw/pilot_01_slow.mp4 \
    --partition-x 320 \
    --zone-width 80 \
    --direction left_to_right \
    --save-output videos/annotated/pilot_01_slow_annotated.mp4
```

Suggested arguments:

| Argument | Description |
|---|---|
| `--video` | Path to input video |
| `--partition-x` | x-coordinate of partition line |
| `--zone-width` | width of crossing detection zone |
| `--direction` | `left_to_right` or `right_to_left` |
| `--min-area` | minimum contour area |
| `--cooldown-frames` | cooldown after each count |
| `--save-output` | path to save annotated video |
| `--display` | show live annotated video window |

---

## 13. Evaluation Metrics

For each video:

```text
absolute_error = abs(automated_count - manual_count)
percent_error = absolute_error / manual_count * 100
```

Recommended summary metrics:

- Mean absolute error
- Median absolute error
- Percent error
- Number of false positives
- Number of missed transfers
- Manual count vs automated count plot

Recommended results file:

```text
results/feasibility_summary.csv
```

Example format:

```csv
video_id,manual_count,automated_count,absolute_error,percent_error,condition,main_failure_mode
pilot_01,10,10,0,0.0,slow ideal lighting,none
pilot_02,20,18,2,10.0,normal ideal lighting,missed fast transfer
pilot_03,20,24,4,20.0,fast movement,double counting
pilot_04,20,15,5,25.0,hand occlusion,missed occluded block
pilot_05,55,49,6,10.9,full trial,mixed errors
```

---

## 14. Feasibility Decision Criteria

Use the pilot results to decide whether this approach is appropriate for the larger student project.

### Green Light

Proceed with the 6-week student project if:

- The algorithm works under ideal controlled videos.
- Average error is reasonably low.
- Most errors are explainable.
- The setup is easy to reproduce.
- Improvements can be made with simple logic.

Suggested benchmark:

```text
Ideal videos: within ±1–2 blocks
Controlled normal videos: within ±5 blocks
Full 60-second trial: within approximately ±10% error
```

### Yellow Light

Proceed as a **semi-automated scoring aid** if:

- The algorithm detects most transfers.
- Manual correction is still needed.
- Errors mostly come from hand occlusion or double-counting.
- The system is useful for video review but not fully automatic scoring.

Possible revised project title:

```text
Semi-Automated Video-Based Review of Box and Block Test Performance
```

### Red Light

Do not use crossing-line detection as the main student project if:

- The algorithm fails even under ideal conditions.
- Counts are highly unstable across repeated tests.
- Hand occlusion makes transfer detection unreliable.
- The project would require advanced deep learning before any result is possible.

---

## 15. Common Failure Cases

Document failure cases carefully. These are useful research findings.

| Failure Case | Likely Effect | Possible Fix |
|---|---|---|
| Hand crosses partition without a successful block transfer | False positive | Add target-side confirmation |
| Block hidden by hand | Missed transfer | Improve camera angle |
| Hand moves back and forth near partition | Double count | Increase cooldown period |
| Two blocks moved together | Incorrect score | Document limitation |
| Fast hand movement | Missed count | Reduce blur; increase frame rate |
| Poor lighting | Noisy motion mask | Improve lighting |
| Camera movement | False motion | Use tripod/fixed mount |
| Box shifts during trial | Misaligned partition line | Tape box position |

---

## 16. Recommended Improvements After Initial Testing

If the simple prototype works, consider adding:

### Direction-Specific Counting

Only count movement in the valid transfer direction.

```text
right-hand trial: left → right
left-hand trial: right → left
```

### Cooldown Period

Ignore new events for a short period after each count.

Example:

```text
0.5–1.0 seconds
```

### Region Restriction

Only analyze motion near the partition and inside the box area.

### Target-Side Confirmation

Count only if motion crossing the partition is followed by activity or a new object-like region in the target compartment.

### Interactive ROI Selection

Allow the user to select the partition line and box area manually at the beginning of the video.

### Annotated Output Video

Save a video showing:

- Partition line
- Crossing zone
- Detected motion
- Current count
- Frame number
- Counted event marker

---

## 17. Suggested Student-Friendly Research Framing

If feasibility is promising, this can become a student summer research project framed as:

> This project develops and evaluates a simple computer vision prototype for estimating Box and Block Test scores from fixed-camera video recordings.

The student can focus on:

- Recording standardized videos.
- Manually scoring ground truth.
- Improving crossing-line detection.
- Quantifying automated scoring error.
- Identifying failure cases.
- Presenting whether the method is feasible.

---

## 18. Ethical and Practical Notes

For early feasibility testing:

- Use simulated BBT trials with healthy volunteers or lab members.
- Avoid collecting patient data unless appropriate approval is in place.
- Avoid recording identifiable faces if not needed.
- Store videos securely.
- Use videos only for development and educational research purposes.
- Make clear that the tool is not intended for clinical decision-making.

---

## 19. Final Feasibility Output

At the end of this feasibility test, produce:

1. A short technical summary.
2. A table comparing manual and automated counts.
3. Annotated example videos.
4. A list of failure cases.
5. A recommendation:

```text
Green light: suitable for 6-week student project
Yellow light: suitable if framed as semi-automated video review
Red light: not suitable without advanced methods
```

---

## 20. Suggested Next Steps

1. Record 3–5 pilot videos.
2. Run the minimal crossing-line script.
3. Tune the partition line, crossing zone width, minimum area, and cooldown.
4. Compare automated count with manual count.
5. Decide whether to proceed with the full student project.

---

## 21. Project Status

Current status:

```text
Stage: Technical feasibility testing
Approach: Option A — crossing-line detection
Primary method: OpenCV motion detection near partition
Validation: Manual count comparison
Clinical readiness: Not clinical-ready; feasibility only
```
