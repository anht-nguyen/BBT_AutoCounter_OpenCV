from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bbt_autocounter.hand_confirmation import HandMotionConfirmator
from bbt_autocounter.motion import MotionMaskCleanerConfig, clean_motion_mask
from bbt_autocounter.ui import resize_frame


FEASIBILITY_ROOT = PROJECT_ROOT / "feasibility"
VIDEO_PATH = FEASIBILITY_ROOT / "data" / "videos" / "raw" / "BBT-ground_truth.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MediaPipe-based hand confirmation on a raw BBT video.")
    parser.add_argument("--video", type=Path, default=VIDEO_PATH)
    parser.add_argument("--model-asset-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    confirmator = HandMotionConfirmator(partition_x=640, direction="left_to_right", fingertip_margin=5, hand_mask_padding=20, hand_mask_dilation=15, selected_fingertips=("thumb", "index", "middle"), model_asset_path=args.model_asset_path)
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
            cleaned_motion_mask = clean_motion_mask(subtractor.apply(resized), config=cleaner)
            result = confirmator.analyze_frame(resized, cleaned_motion_mask)
            overlay = confirmator.draw_debug_overlay(resized, result, target_non_hand_motion_threshold=300)
            cv2.imshow("Hand Confirmation Overlay", overlay)
            cv2.imshow("Motion Mask", cleaned_motion_mask)
            cv2.imshow("Hand Region Mask", result.hand_region_mask if result.hand_region_mask is not None else np.zeros_like(cleaned_motion_mask))
            cv2.imshow("Hand Motion Mask", result.hand_motion_mask if result.hand_motion_mask is not None else np.zeros_like(cleaned_motion_mask))
            cv2.imshow("Non-Hand Motion Mask", result.non_hand_motion_mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                cv2.waitKey(0)
    finally:
        capture.release()
        confirmator.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
