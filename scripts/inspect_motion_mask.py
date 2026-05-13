from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bbt_autocounter.motion import MotionMaskCleanerConfig, clean_motion_mask, draw_contour_debug, filter_contours_by_area
from bbt_autocounter.ui import add_panel_label, mask_to_bgr, resize_for_display, resize_frame, stack_views


FEASIBILITY_ROOT = PROJECT_ROOT / "feasibility"
VIDEO_PATH = FEASIBILITY_ROOT / "data" / "videos" / "raw" / "BBT-ground_truth.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect motion mask cleaning on a raw BBT video.")
    parser.add_argument("--video", type=Path, default=VIDEO_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    cleaner = MotionMaskCleanerConfig()
    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break
            resized = resize_frame(frame, resize_width=1280, resize_height=720)
            raw_mask = subtractor.apply(resized)
            cleaned_mask = clean_motion_mask(raw_mask, config=cleaner)
            contours = filter_contours_by_area(cleaned_mask, min_area=500)
            contour_debug = draw_contour_debug(resized, contours)
            comparison = stack_views(
                [
                    [add_panel_label(resized, "Original Frame"), add_panel_label(mask_to_bgr(raw_mask), "Raw Motion Mask")],
                    [add_panel_label(mask_to_bgr(cleaned_mask), "Cleaned Motion Mask"), add_panel_label(contour_debug, "Contour Debug")],
                ]
            )
            cv2.imshow("BBT Motion Mask Cleaning Comparison", resize_for_display(comparison, 0.60))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                cv2.waitKey(0)
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
