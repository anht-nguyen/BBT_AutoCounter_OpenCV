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
from bbt_autocounter.hand_confirmation import HandMotionConfirmator
from bbt_autocounter.motion import clean_motion_mask
from bbt_autocounter.motion import draw_contour_debug
from bbt_autocounter.motion import filter_contours_by_area
from bbt_autocounter.object_confirmation import ObjectTransferConfirmator
from bbt_autocounter.ui import resize_frame


VIDEO_PATH = FEASIBILITY_ROOT / "data" / "videos" / "raw" / "BBT-ground_truth.mp4"
OUTPUT_DIR = FEASIBILITY_ROOT / "data" / "videos" / "annotated"
WINDOW_NAME = "BBT Scoring With Hand Confirmation"
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

FINGERTIP_MARGIN = 5
HAND_MASK_PADDING = 20
HAND_MASK_DILATION = 15
TARGET_NON_HAND_MOTION_THRESHOLD = 300
PERSISTENCE_MOTION_THRESHOLD = 120
PERSISTENCE_FRAMES_REQUIRED = 2
ABSENCE_RESET_FRAMES = 4
SELECTED_FINGERTIPS = ("thumb", "index", "middle")
MODEL_ASSET_PATH = None

# A short memory helps align fingertip crossing with the motion-based event.
FINGERTIP_CONFIRM_WINDOW_FRAMES = 10
OBJECT_CONFIRM_WINDOW_FRAMES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect BBT scoring with both motion crossing and hand confirmation."
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
    parser.add_argument(
        "--save-output",
        type=Path,
        default=None,
        help="Optional path for the saved annotated comparison video.",
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


def stack_views(
    original_frame: np.ndarray,
    cleaned_mask: np.ndarray,
    contour_debug_frame: np.ndarray,
    scoring_frame: np.ndarray,
) -> np.ndarray:
    top_row = np.hstack(
        [
            add_panel_label(original_frame, "Original Frame"),
            add_panel_label(mask_to_bgr(cleaned_mask), "Cleaned Motion Mask"),
        ]
    )
    bottom_row = np.hstack(
        [
            add_panel_label(contour_debug_frame, "Contour Debug"),
            add_panel_label(scoring_frame, "Scoring And Count"),
        ]
    )
    return np.vstack([top_row, bottom_row])


def open_writer(output_path: Path, frame_size: tuple[int, int], fps: float) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {output_path}")
    return writer


def resolve_output_path(video_path: Path, save_output: Path | None) -> Path:
    if save_output is not None:
        return save_output.expanduser().resolve()
    return (OUTPUT_DIR / f"{video_path.stem}_scoring_with_hand_confirmation.mp4").resolve()


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    output_path = resolve_output_path(video_path, args.save_output)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

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
        capture.release()
        raise RuntimeError(
            f"{exc}\n\n"
            "To fix this, either:\n"
            "1. pass --model-asset-path path/to/hand_landmarker.task\n"
            "2. or place the model at feasibility/models/hand_landmarker.task"
        ) from exc

    object_confirmator = ObjectTransferConfirmator(
        partition_x=PARTITION_X,
        direction=DIRECTION,
        target_motion_threshold=TARGET_NON_HAND_MOTION_THRESHOLD,
        persistence_motion_threshold=PERSISTENCE_MOTION_THRESHOLD,
        persistence_frames_required=PERSISTENCE_FRAMES_REQUIRED,
        absence_reset_frames=ABSENCE_RESET_FRAMES,
    )

    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=False,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    writer = None
    frame_index = 0
    combined_count = 0
    last_fingertip_cross_frame = None
    last_object_confirm_frame = None
    total_motion_events = 0
    total_hand_confirmed_events = 0
    rejected_motion_events_without_hand = 0
    rejected_motion_events_without_object = 0

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

            motion_result = counter.update(cleaned_mask, frame_idx=frame_index)
            hand_result = confirmator.analyze_frame(resized_frame, cleaned_mask)
            object_result = object_confirmator.analyze_frame(hand_result["non_hand_motion_mask"])

            if hand_result["fingertip_crossed"]:
                last_fingertip_cross_frame = frame_index
            if object_result.object_confirmed:
                last_object_confirm_frame = frame_index

            fingertip_recent = (
                last_fingertip_cross_frame is not None
                and (frame_index - last_fingertip_cross_frame) <= FINGERTIP_CONFIRM_WINDOW_FRAMES
            )
            object_recent = (
                last_object_confirm_frame is not None
                and (frame_index - last_object_confirm_frame) <= OBJECT_CONFIRM_WINDOW_FRAMES
            )
            possible_block_without_fingertip = confirmator.detect_block_without_fingertip_crossing(
                hand_result,
                threshold=TARGET_NON_HAND_MOTION_THRESHOLD,
            )

            motion_event_detected = bool(motion_result["event_detected"])
            hand_confirmed_score = motion_event_detected and fingertip_recent and object_recent
            if motion_event_detected:
                total_motion_events += 1
                if hand_confirmed_score:
                    combined_count += 1
                    total_hand_confirmed_events += 1
                else:
                    if not fingertip_recent:
                        rejected_motion_events_without_hand += 1
                    if not object_recent:
                        rejected_motion_events_without_object += 1

            scoring_frame = confirmator.draw_debug_overlay(
                resized_frame,
                hand_result,
                target_non_hand_motion_threshold=TARGET_NON_HAND_MOTION_THRESHOLD,
            )
            scoring_frame = object_confirmator.draw_debug_overlay(scoring_frame, object_result)
            scoring_frame = counter.draw_debug_overlay(scoring_frame, motion_result)

            score_label = f"Scored: {'YES' if hand_confirmed_score else 'NO'}"
            count_label = f"Count: {combined_count}"
            status_label = f"Motion status: {motion_result['score_status']}"
            mode_label = f"Motion mode: {motion_result['confirmation_mode']} | State: {motion_result['state']}"
            hand_label = (
                f"Fingertip crossed: {'YES' if hand_result['fingertip_crossed'] else 'NO'} | "
                f"Recent: {'YES' if fingertip_recent else 'NO'}"
            )
            object_label = (
                f"Object confirmed: {'YES' if object_result.object_confirmed else 'NO'} | "
                f"Recent: {'YES' if object_recent else 'NO'}"
            )
            invalid_label = (
                f"Possible block without fingertip: {'YES' if possible_block_without_fingertip else 'NO'} | "
                f"Target non-hand area: {hand_result['target_non_hand_motion_area']}"
            )
            event_label = (
                f"Motion event: {'YES' if motion_event_detected else 'NO'} | "
                f"Hand-confirmed events: {total_hand_confirmed_events} | "
                f"Rejected by hand gate: {rejected_motion_events_without_hand}"
            )
            object_event_label = (
                f"Persistence: {'YES' if object_result.persistence_detected else 'NO'} | "
                f"Object gate rejects: {rejected_motion_events_without_object}"
            )

            cv2.putText(
                scoring_frame,
                f"{score_label} | {count_label}",
                (20, resized_frame.shape[0] - 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.82,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                scoring_frame,
                f"{status_label} | {mode_label}",
                (20, resized_frame.shape[0] - 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                scoring_frame,
                hand_label,
                (20, resized_frame.shape[0] - 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                scoring_frame,
                object_label,
                (20, resized_frame.shape[0] - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                scoring_frame,
                invalid_label,
                (20, resized_frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                scoring_frame,
                event_label,
                (20, resized_frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                scoring_frame,
                object_event_label,
                (20, resized_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if hand_confirmed_score:
                cv2.putText(
                    scoring_frame,
                    "HAND-CONFIRMED SCORE!",
                    (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 120),
                    3,
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
                original_frame=resized_frame,
                cleaned_mask=cleaned_mask,
                contour_debug_frame=contour_debug_frame,
                scoring_frame=scoring_frame,
            )

            if writer is None:
                writer = open_writer(
                    output_path,
                    (comparison_view.shape[1], comparison_view.shape[0]),
                    fps,
                )

            writer.write(comparison_view)
            cv2.imshow(WINDOW_NAME, resize_for_display(comparison_view, DISPLAY_SCALE))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                cv2.waitKey(0)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        confirmator.close()
        cv2.destroyAllWindows()

    print(f"Processed video: {video_path}")
    print(f"Frames processed: {frame_index}")
    print(f"Saved annotated video: {output_path}")
    print(f"Combined score count: {combined_count}")
    print(f"Total motion events: {total_motion_events}")
    print(f"Hand-confirmed events: {total_hand_confirmed_events}")
    print(f"Rejected motion events without hand confirmation: {rejected_motion_events_without_hand}")
    print(f"Rejected motion events without object confirmation: {rejected_motion_events_without_object}")
    print(f"Motion detector candidate crossings: {counter.total_candidate_crossings}")
    print(f"Motion detector confirmed crossings: {counter.confirmed_crossings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
