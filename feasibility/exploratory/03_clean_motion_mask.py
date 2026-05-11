from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEASIBILITY_ROOT = PROJECT_ROOT / "feasibility"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feasibility.src.motion_mask_cleaning import clean_motion_mask
from feasibility.src.motion_mask_cleaning import draw_contour_debug
from feasibility.src.motion_mask_cleaning import filter_contours_by_area
from feasibility.src.motion_mask_cleaning import resize_frame


VIDEO_PATH = FEASIBILITY_ROOT / "data" / "videos" / "raw" / "BBT-ground_truth.mp4"
WINDOW_NAME = "BBT Motion Mask Cleaning Comparison"
DISPLAY_SCALE = 0.60
RESIZE_WIDTH = 1280
RESIZE_HEIGHT = 720

BLUR_KERNEL_SIZE = 5
THRESHOLD_VALUE = 200
MORPH_KERNEL_SIZE = 5
OPENING_ITERATIONS = 1
CLOSING_ITERATIONS = 1
MIN_AREA = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect motion mask cleaning on a raw BBT video."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=VIDEO_PATH,
        help="Path to the raw input video.",
    )
    return parser.parse_args()


def add_panel_label(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    cv2.putText(labeled, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(labeled, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def resize_for_display(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def stack_debug_views(
    original_frame: np.ndarray,
    raw_mask: np.ndarray,
    cleaned_mask: np.ndarray,
    contour_debug_frame: np.ndarray,
) -> np.ndarray:
    top_row = np.hstack(
        [
            add_panel_label(original_frame, "Original Frame"),
            add_panel_label(mask_to_bgr(raw_mask), "Raw Mask"),
        ]
    )
    bottom_row = np.hstack(
        [
            add_panel_label(mask_to_bgr(cleaned_mask), "Cleaned Mask"),
            add_panel_label(contour_debug_frame, "Contour Debug"),
        ]
    )
    return np.vstack([top_row, bottom_row])


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    # This background subtractor gives us the same kind of raw motion mask used in Step 2.
    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=False,
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

            # Cleaning makes the motion blob easier to inspect than the raw mask alone.
            cleaned_mask = clean_motion_mask(
                raw_mask,
                blur_kernel_size=BLUR_KERNEL_SIZE,
                threshold_value=THRESHOLD_VALUE,
                morph_kernel_size=MORPH_KERNEL_SIZE,
                opening_iterations=OPENING_ITERATIONS,
                closing_iterations=CLOSING_ITERATIONS,
            )
            contour_info_list = filter_contours_by_area(cleaned_mask, min_area=MIN_AREA)
            contour_debug_frame = draw_contour_debug(resized_frame, contour_info_list)

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

            comparison_view = stack_debug_views(
                original_frame=resized_frame,
                raw_mask=raw_mask,
                cleaned_mask=cleaned_mask,
                contour_debug_frame=contour_debug_frame,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
