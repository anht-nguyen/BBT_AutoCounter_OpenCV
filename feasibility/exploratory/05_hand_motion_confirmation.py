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

from bbt_autocounter.hand_confirmation import HandMotionConfirmator
from bbt_autocounter.motion import clean_motion_mask
from bbt_autocounter.ui import resize_frame


VIDEO_PATH = FEASIBILITY_ROOT / "data" / "videos" / "raw" / "BBT-ground_truth.mp4"
WINDOW_NAME = "BBT Hand Motion Confirmation"
DISPLAY_SCALE = 0.60
RESIZE_WIDTH = 1280
RESIZE_HEIGHT = 720

PARTITION_X = 640
DIRECTION = "left_to_right"
FINGERTIP_MARGIN = 10
HAND_MASK_PADDING = 20
HAND_MASK_DILATION = 15
TARGET_NON_HAND_MOTION_THRESHOLD = 300
SELECTED_FINGERTIPS = ("thumb", "index", "middle")
MODEL_ASSET_PATH = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect MediaPipe-based hand confirmation on the BBT pilot video."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=VIDEO_PATH,
        help="Path to the raw input video.",
    )
    parser.add_argument(
        "--model-asset-path",
        type=Path,
        default=MODEL_ASSET_PATH,
        help="Optional MediaPipe Hand Landmarker .task model path for Tasks-only installs.",
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


def combine_mask_debug_view(
    cleaned_motion_mask: np.ndarray,
    hand_motion_mask: np.ndarray | None,
    non_hand_motion_mask: np.ndarray | None,
) -> np.ndarray:
    cleaned_panel = mask_to_bgr(cleaned_motion_mask)
    hand_panel = mask_to_bgr(hand_motion_mask if hand_motion_mask is not None else np.zeros_like(cleaned_motion_mask))
    non_hand_panel = mask_to_bgr(
        non_hand_motion_mask if non_hand_motion_mask is not None else np.zeros_like(cleaned_motion_mask)
    )

    top_strip = np.hstack(
        [
            add_panel_label(cleaned_panel, "Cleaned Motion"),
            add_panel_label(hand_panel, "Hand Motion"),
        ]
    )
    bottom_strip = np.hstack(
        [
            add_panel_label(non_hand_panel, "Non-Hand Motion"),
            add_panel_label(non_hand_panel, "Target-Side Non-Hand Motion"),
        ]
    )
    return np.vstack([top_strip, bottom_strip])


def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] == target_height:
        return image
    scale = float(target_height) / float(image.shape[0])
    target_width = int(round(image.shape[1] * scale))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def stack_views(
    original_frame: np.ndarray,
    cleaned_motion_mask: np.ndarray,
    mask_debug_frame: np.ndarray,
    overlay_frame: np.ndarray,
) -> np.ndarray:
    mask_debug_frame = resize_to_height(mask_debug_frame, overlay_frame.shape[0])
    top_row = np.hstack(
        [
            add_panel_label(original_frame, "Original Frame"),
            add_panel_label(mask_to_bgr(cleaned_motion_mask), "Cleaned Motion Mask"),
        ]
    )
    bottom_row = np.hstack(
        [
            add_panel_label(mask_debug_frame, "Hand vs Non-Hand Motion"),
            add_panel_label(overlay_frame, "Hand Confirmation Overlay"),
        ]
    )
    return np.vstack([top_row, bottom_row])


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()

    try:
        confirmator = HandMotionConfirmator(
            partition_x=PARTITION_X,
            direction=DIRECTION,
            fingertip_margin=FINGERTIP_MARGIN,
            hand_mask_padding=HAND_MASK_PADDING,
            hand_mask_dilation=HAND_MASK_DILATION,
            selected_fingertips=SELECTED_FINGERTIPS,
            model_asset_path=args.model_asset_path,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}\n\n"
            "To fix this, either:\n"
            "1. pass --model-asset-path path/to/hand_landmarker.task\n"
            "2. or place the model at feasibility/models/hand_landmarker.task"
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        confirmator.close()
        raise FileNotFoundError(f"Could not open video: {video_path}")

    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=False,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    total_frames = 0
    frames_with_hand_detected = 0
    frames_with_fingertip_crossing = 0
    frames_flagged_possible_block_without_fingertip = 0

    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break

            total_frames += 1
            resized_frame = resize_frame(
                frame,
                resize_width=RESIZE_WIDTH,
                resize_height=RESIZE_HEIGHT,
            )
            raw_motion_mask = background_subtractor.apply(resized_frame)
            cleaned_motion_mask = clean_motion_mask(raw_motion_mask)
            result = confirmator.analyze_frame(resized_frame, cleaned_motion_mask)
            overlay_frame = confirmator.draw_debug_overlay(
                resized_frame,
                result,
                target_non_hand_motion_threshold=TARGET_NON_HAND_MOTION_THRESHOLD,
            )

            if result["hand_detected"]:
                frames_with_hand_detected += 1
            if result["fingertip_crossed"]:
                frames_with_fingertip_crossing += 1

            possible_block_without_fingertip = confirmator.detect_block_without_fingertip_crossing(
                result,
                threshold=TARGET_NON_HAND_MOTION_THRESHOLD,
            )
            if possible_block_without_fingertip:
                frames_flagged_possible_block_without_fingertip += 1

            mask_debug_frame = combine_mask_debug_view(
                cleaned_motion_mask=cleaned_motion_mask,
                hand_motion_mask=result["hand_motion_mask"],
                non_hand_motion_mask=result["non_hand_motion_mask"],
            )

            cv2.putText(
                overlay_frame,
                f"Frames: {total_frames} | Hand detected: {'YES' if result['hand_detected'] else 'NO'}",
                (20, resized_frame.shape[0] - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay_frame,
                f"Fingertip crossed: {'YES' if result['fingertip_crossed'] else 'NO'} | "
                f"Target non-hand area: {result['target_non_hand_motion_area']}",
                (20, resized_frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay_frame,
                f"Possible block without fingertip: {'YES' if possible_block_without_fingertip else 'NO'}",
                (20, resized_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            comparison_view = stack_views(
                original_frame=resized_frame,
                cleaned_motion_mask=cleaned_motion_mask,
                mask_debug_frame=mask_debug_frame,
                overlay_frame=overlay_frame,
            )
            cv2.imshow(WINDOW_NAME, resize_for_display(comparison_view, DISPLAY_SCALE))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                cv2.waitKey(0)
    finally:
        capture.release()
        confirmator.close()
        cv2.destroyAllWindows()

    print(f"Processed video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Frames with hand detected: {frames_with_hand_detected}")
    print(f"Frames with fingertip crossing: {frames_with_fingertip_crossing}")
    print(
        "Frames flagged as possible block-without-fingertip-crossing: "
        f"{frames_flagged_possible_block_without_fingertip}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
