from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEASIBILITY_ROOT = PROJECT_ROOT / "feasibility"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bbt_autocounter.crossing import CrossingCounter
from bbt_autocounter.motion import clean_motion_mask
from bbt_autocounter.motion import draw_contour_debug
from bbt_autocounter.motion import filter_contours_by_area
from bbt_autocounter.ui import resize_frame


VIDEO_PATH = FEASIBILITY_ROOT / "data" / "videos" / "raw" / "BBT-ground_truth.mp4"
WINDOW_NAME = "BBT Crossing Event Detection"
DISPLAY_SCALE = 0.60
RESIZE_WIDTH = 1280
RESIZE_HEIGHT = 720

PARTITION_X = 640
DIRECTION = "left_to_right"
CROSSING_ZONE_WIDTH = 100
DEAD_ZONE_WIDTH = 20
MIN_AREA = 500
COOLDOWN_FRAMES = 20
CONFIRMATION_MODE = "hybrid"
LEADING_EDGE_MARGIN = 10
TARGET_CONFIRMATION_WINDOW_FRAMES = 10
TARGET_MOTION_AREA_THRESHOLD = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect exploratory crossing event detection on a raw BBT video."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=VIDEO_PATH,
        help="Path to the raw input video.",
    )
    return parser.parse_args()


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def add_panel_label(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    cv2.putText(labeled, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(labeled, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled


def resize_for_display(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def stack_views(
    original_frame: np.ndarray,
    cleaned_mask: np.ndarray,
    contour_debug_frame: np.ndarray,
    annotated_frame: np.ndarray,
) -> np.ndarray:
    cleaned_mask_bgr = mask_to_bgr(cleaned_mask)
    top_row = np.hstack(
        [
            add_panel_label(original_frame, "Original Frame"),
            add_panel_label(cleaned_mask_bgr, "Cleaned Motion Mask"),
        ]
    )
    bottom_row = np.hstack(
        [
            add_panel_label(contour_debug_frame, "Contour Debug"),
            add_panel_label(annotated_frame, "Scoring And Count"),
        ]
    )
    return np.vstack([top_row, bottom_row])


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=False,
    )
    counter = CrossingCounter(
        partition_x=PARTITION_X,
        direction=DIRECTION,
        crossing_zone_width=CROSSING_ZONE_WIDTH,
        dead_zone_width=DEAD_ZONE_WIDTH,
        min_area=MIN_AREA,
        cooldown_frames=COOLDOWN_FRAMES,
        leading_edge_margin=LEADING_EDGE_MARGIN,
        confirmation_mode=CONFIRMATION_MODE,
        target_confirmation_window_frames=TARGET_CONFIRMATION_WINDOW_FRAMES,
        target_motion_area_threshold=TARGET_MOTION_AREA_THRESHOLD,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break

            frame_index += 1
            resized_frame = resize_frame(
                frame,
                resize_width=RESIZE_WIDTH,
                resize_height=RESIZE_HEIGHT,
            )
            raw_mask = background_subtractor.apply(resized_frame)
            cleaned_mask = clean_motion_mask(raw_mask)
            contour_info_list = filter_contours_by_area(cleaned_mask, min_area=MIN_AREA)
            contour_debug_frame = draw_contour_debug(resized_frame, contour_info_list)
            result = counter.update(cleaned_mask, frame_idx=frame_index)
            annotated_frame = counter.draw_debug_overlay(resized_frame, result)

            score_label = f"Scored: {'YES' if result['motion_scored'] else 'NO'}"
            count_label = f"Count: {result['count']}"
            status_label = f"Status: {result['score_status']}"
            method_label = f"Mode: {result['confirmation_mode']} | State: {result['state']}"
            edge_label = (
                f"Edges: left={result['left_edge']}, right={result['right_edge']} | "
                f"margin={result['leading_edge_margin']}"
            )
            candidate_label = (
                f"Candidate: {'YES' if result['candidate_pending'] else 'NO'} | "
                f"Age: {result['candidate_age_frames']} | Target confirmed: {'YES' if result['target_confirmed'] else 'NO'}"
            )
            target_area_label = (
                f"Target motion area: {result['target_motion_area']} | "
                f"Cooldown: {result['cooldown_remaining']} | Armed: {'YES' if result['armed_for_crossing'] else 'NO'}"
            )
            cv2.putText(
                annotated_frame,
                f"{score_label} | {count_label}",
                (20, resized_frame.shape[0] - 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                f"{status_label} | {method_label}",
                (20, resized_frame.shape[0] - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                candidate_label,
                (20, resized_frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                f"{edge_label} | {target_area_label}",
                (20, resized_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                contour_debug_frame,
                f"Frame: {frame_index} | Contours kept: {len(contour_info_list)}",
                (20, resized_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            comparison_view = stack_views(
                resized_frame,
                cleaned_mask,
                contour_debug_frame,
                annotated_frame,
            )

            cv2.imshow(WINDOW_NAME, resize_for_display(comparison_view, DISPLAY_SCALE))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                cv2.waitKey(0)
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"Processed video: {video_path}")
    print(f"Frames processed: {frame_index}")
    print(f"Automated crossing count: {counter.count}")
    print(f"Confirmation mode: {counter.confirmation_mode}")
    print(f"Leading edge margin: {counter.leading_edge_margin}")
    print(f"Total candidate crossings: {counter.total_candidate_crossings}")
    print(f"Confirmed crossings: {counter.confirmed_crossings}")
    print(f"Rejected candidates: {counter.rejected_candidates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
