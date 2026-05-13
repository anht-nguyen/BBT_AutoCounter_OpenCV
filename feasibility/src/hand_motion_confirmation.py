from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

try:
    from feasibility.src.motion_mask_cleaning import clean_motion_mask
except ImportError:
    try:
        from motion_mask_cleaning import clean_motion_mask
    except ImportError:
        def clean_motion_mask(
            raw_mask,
            blur_kernel_size=5,
            threshold_value=200,
            morph_kernel_size=5,
            opening_iterations=1,
            closing_iterations=1,
        ):
            """Fallback cleaner used only if the shared mask-cleaning helper is unavailable."""
            blur_kernel_size = max(int(blur_kernel_size), 1)
            morph_kernel_size = max(int(morph_kernel_size), 1)
            if blur_kernel_size % 2 == 0:
                blur_kernel_size += 1
            if morph_kernel_size % 2 == 0:
                morph_kernel_size += 1

            blurred_mask = cv2.GaussianBlur(raw_mask, (blur_kernel_size, blur_kernel_size), 0)
            _, binary_mask = cv2.threshold(blurred_mask, int(threshold_value), 255, cv2.THRESH_BINARY)
            kernel = np.ones((morph_kernel_size, morph_kernel_size), dtype=np.uint8)
            opened_mask = cv2.morphologyEx(
                binary_mask,
                cv2.MORPH_OPEN,
                kernel,
                iterations=max(int(opening_iterations), 0),
            )
            cleaned_mask = cv2.morphologyEx(
                opened_mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=max(int(closing_iterations), 0),
            )
            return cleaned_mask


FINGERTIP_NAME_TO_INDEX = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
WRIST_INDEX = 0
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CANDIDATE_PATHS = (
    PROJECT_ROOT / "feasibility" / "models" / "hand_landmarker.task",
    PROJECT_ROOT / "models" / "hand_landmarker.task",
    PROJECT_ROOT / "feasibility" / "models" / "mediapipe" / "hand_landmarker.task",
)


@dataclass
class LandmarkDetectionResult:
    hand_detected: bool
    landmarks: list[tuple[int, int]] | None
    handedness: str | None


def resize_frame(frame, resize_width=None, resize_height=None):
    if resize_width is None and resize_height is None:
        return frame

    height, width = frame.shape[:2]
    target_width = resize_width
    target_height = resize_height
    if target_width is None:
        if target_height is None:
            raise ValueError("resize_height must be provided when resize_width is omitted")
        scale = float(target_height) / float(height)
        target_width = int(round(width * scale))
    elif target_height is None:
        scale = float(target_width) / float(width)
        target_height = int(round(height * scale))

    return cv2.resize(frame, (int(target_width), int(target_height)), interpolation=cv2.INTER_AREA)


def mask_to_bgr(mask):
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


class HandMotionConfirmator:
    """Approximate hand/fingertip confirmation using MediaPipe landmarks.

    The hand-region mask created here is not a true segmentation mask.
    It is only an approximate region built from detected hand landmarks.
    """

    def __init__(
        self,
        partition_x: int,
        direction: str = "left_to_right",
        fingertip_margin: int = 10,
        hand_mask_padding: int = 20,
        hand_mask_dilation: int = 15,
        selected_fingertips=("thumb", "index", "middle"),
        require_all_selected_fingertips: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_asset_path: str | Path | None = None,
    ) -> None:
        self.partition_x = int(partition_x)
        self.direction = str(direction)
        self.fingertip_margin = int(fingertip_margin)
        self.hand_mask_padding = int(hand_mask_padding)
        self.hand_mask_dilation = int(hand_mask_dilation)
        self.selected_fingertips = tuple(selected_fingertips)
        self.require_all_selected_fingertips = bool(require_all_selected_fingertips)
        self.max_num_hands = max(int(max_num_hands), 1)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.model_asset_path = None if model_asset_path is None else Path(model_asset_path)

        self._validate_inputs()
        self._mp, self._hands, self._backend = self._create_mediapipe_hands()

    def _validate_inputs(self) -> None:
        if self.direction not in {"left_to_right", "right_to_left"}:
            raise ValueError("direction must be 'left_to_right' or 'right_to_left'")
        if self.fingertip_margin < 0 or self.hand_mask_padding < 0 or self.hand_mask_dilation < 0:
            raise ValueError("margins, padding, and dilation must be nonnegative integers")
        if not 0.0 <= self.min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be between 0 and 1")
        if not 0.0 <= self.min_tracking_confidence <= 1.0:
            raise ValueError("min_tracking_confidence must be between 0 and 1")
        for fingertip_name in self.selected_fingertips:
            if fingertip_name not in FINGERTIP_NAME_TO_INDEX:
                raise ValueError(
                    f"Unsupported fingertip '{fingertip_name}'. "
                    f"Choose from: {', '.join(FINGERTIP_NAME_TO_INDEX)}"
                )

    @staticmethod
    def _load_mediapipe() -> Any:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "MediaPipe is required for hand confirmation. Install with: pip install mediapipe"
            ) from exc
        return mp

    def _resolve_model_asset_path(self) -> Path | None:
        if self.model_asset_path is not None:
            return self.model_asset_path.expanduser().resolve()

        for candidate_path in DEFAULT_MODEL_CANDIDATE_PATHS:
            if candidate_path.exists():
                return candidate_path.resolve()
        return None

    def _create_mediapipe_hands(self) -> tuple[Any, Any, Literal["solutions", "tasks"]]:
        mp = self._load_mediapipe()
        solutions = getattr(mp, "solutions", None)
        hands_module = getattr(solutions, "hands", None)
        if hands_module is not None:
            hands = hands_module.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            return mp, hands, "solutions"

        if hasattr(mp, "tasks"):
            resolved_model_path = self._resolve_model_asset_path()
            if resolved_model_path is None:
                searched_paths = "\n".join(f"- {path}" for path in DEFAULT_MODEL_CANDIDATE_PATHS)
                raise RuntimeError(
                    "This MediaPipe installation exposes only the Tasks API. "
                    "Provide a Hand Landmarker model with model_asset_path "
                    "or --model-asset-path.\n"
                    "Searched default paths:\n"
                    f"{searched_paths}"
                )
            if not resolved_model_path.exists():
                raise FileNotFoundError(
                    f"Could not find MediaPipe hand landmarker model: {resolved_model_path}"
                )

            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision

            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(resolved_model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                min_hand_presence_confidence=self.min_tracking_confidence,
            )
            hands = vision.HandLandmarker.create_from_options(options)
            self.model_asset_path = resolved_model_path
            return mp, hands, "tasks"

        raise RuntimeError("Unsupported MediaPipe installation: no supported hand detection API found.")

    def detect_landmarks(self, frame) -> LandmarkDetectionResult:
        """Detect one hand and return landmark pixel coordinates."""
        frame_height, frame_width = frame.shape[:2]

        if self._backend == "solutions":
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._hands.process(frame_rgb)

            if not results.multi_hand_landmarks:
                return LandmarkDetectionResult(hand_detected=False, landmarks=None, handedness=None)

            hand_landmarks = results.multi_hand_landmarks[0]
            pixel_landmarks: list[tuple[int, int]] = []
            for landmark in hand_landmarks.landmark:
                x = int(round(landmark.x * frame_width))
                y = int(round(landmark.y * frame_height))
                x = int(np.clip(x, 0, frame_width - 1))
                y = int(np.clip(y, 0, frame_height - 1))
                pixel_landmarks.append((x, y))

            handedness = None
            if results.multi_handedness:
                handedness = results.multi_handedness[0].classification[0].label

            return LandmarkDetectionResult(hand_detected=True, landmarks=pixel_landmarks, handedness=handedness)

        if self._backend == "tasks":
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = self._hands.detect(mp_image)
            if not results.hand_landmarks:
                return LandmarkDetectionResult(hand_detected=False, landmarks=None, handedness=None)

            hand_landmarks = results.hand_landmarks[0]
            pixel_landmarks: list[tuple[int, int]] = []
            for landmark in hand_landmarks:
                x = int(round(landmark.x * frame_width))
                y = int(round(landmark.y * frame_height))
                x = int(np.clip(x, 0, frame_width - 1))
                y = int(np.clip(y, 0, frame_height - 1))
                pixel_landmarks.append((x, y))

            handedness = None
            if getattr(results, "handedness", None):
                handedness = results.handedness[0][0].category_name

            return LandmarkDetectionResult(hand_detected=True, landmarks=pixel_landmarks, handedness=handedness)

        raise RuntimeError(f"Unsupported MediaPipe backend: {self._backend}")

    def create_hand_region_mask(self, frame_shape, landmarks):
        """Build an approximate hand mask from landmarks using a convex hull.

        This is only a rough region around the hand. It is not a true segmentation mask.
        """
        frame_height, frame_width = frame_shape[:2]
        hand_region_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        if not landmarks:
            return hand_region_mask

        points = np.asarray(landmarks, dtype=np.int32)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(hand_region_mask, hull, 255)

        if self.hand_mask_padding > 0:
            x, y, w, h = cv2.boundingRect(points)
            x1 = max(0, x - self.hand_mask_padding)
            y1 = max(0, y - self.hand_mask_padding)
            x2 = min(frame_width, x + w + self.hand_mask_padding)
            y2 = min(frame_height, y + h + self.hand_mask_padding)
            hand_region_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                hand_region_mask[y1:y2, x1:x2],
                np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8),
            )

        if self.hand_mask_dilation > 0:
            kernel_size = max(1, int(self.hand_mask_dilation))
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            hand_region_mask = cv2.dilate(hand_region_mask, kernel, iterations=1)

        return hand_region_mask

    def split_motion_mask(self, motion_mask, hand_region_mask):
        """Separate motion likely caused by the hand from motion outside the hand region."""
        hand_motion_mask = cv2.bitwise_and(motion_mask, hand_region_mask)
        inverse_hand_mask = cv2.bitwise_not(hand_region_mask)
        non_hand_motion_mask = cv2.bitwise_and(motion_mask, inverse_hand_mask)
        return hand_motion_mask, non_hand_motion_mask

    def check_fingertip_crossing(self, landmarks):
        if not landmarks:
            return False, [], {}

        crossed_fingertips = []
        fingertip_positions = {}

        for fingertip_name in self.selected_fingertips:
            landmark_index = FINGERTIP_NAME_TO_INDEX[fingertip_name]
            fingertip_x, fingertip_y = landmarks[landmark_index]
            fingertip_positions[fingertip_name] = (fingertip_x, fingertip_y)

            if self.direction == "left_to_right":
                fingertip_crossed = fingertip_x > (self.partition_x + self.fingertip_margin)
            else:
                fingertip_crossed = fingertip_x < (self.partition_x - self.fingertip_margin)

            if fingertip_crossed:
                crossed_fingertips.append(fingertip_name)

        if self.require_all_selected_fingertips:
            selected_visible = [name for name in self.selected_fingertips if name in fingertip_positions]
            fingertip_crossed = bool(selected_visible) and len(crossed_fingertips) == len(selected_visible)
        else:
            fingertip_crossed = len(crossed_fingertips) > 0

        return fingertip_crossed, crossed_fingertips, fingertip_positions

    def compute_target_side_non_hand_motion_area(self, non_hand_motion_mask):
        frame_width = non_hand_motion_mask.shape[1]
        if self.direction == "left_to_right":
            target_mask = non_hand_motion_mask[:, min(self.partition_x, frame_width):]
        else:
            target_mask = non_hand_motion_mask[:, :max(self.partition_x, 0)]
        return int(cv2.countNonZero(target_mask))

    def analyze_frame(self, frame, motion_mask):
        landmark_result = self.detect_landmarks(frame)

        hand_region_mask = None
        hand_motion_mask = None
        non_hand_motion_mask = motion_mask.copy()
        fingertip_crossed = False
        crossed_fingertips = []
        fingertip_positions = {}

        if landmark_result.hand_detected:
            hand_region_mask = self.create_hand_region_mask(frame.shape, landmark_result.landmarks)
            hand_motion_mask, non_hand_motion_mask = self.split_motion_mask(motion_mask, hand_region_mask)
            fingertip_crossed, crossed_fingertips, fingertip_positions = self.check_fingertip_crossing(
                landmark_result.landmarks
            )

        target_non_hand_motion_area = self.compute_target_side_non_hand_motion_area(non_hand_motion_mask)

        return {
            "hand_detected": landmark_result.hand_detected,
            "landmarks": landmark_result.landmarks,
            "handedness": landmark_result.handedness,
            "fingertip_crossed": fingertip_crossed,
            "crossed_fingertips": crossed_fingertips,
            "fingertip_positions": fingertip_positions,
            "hand_region_mask": hand_region_mask,
            "hand_motion_mask": hand_motion_mask,
            "non_hand_motion_mask": non_hand_motion_mask,
            "target_non_hand_motion_area": target_non_hand_motion_area,
            "direction": self.direction,
            "partition_x": self.partition_x,
            "fingertip_margin": self.fingertip_margin,
        }

    @staticmethod
    def detect_block_without_fingertip_crossing(result, threshold):
        return (
            int(result["target_non_hand_motion_area"]) >= int(threshold)
            and not bool(result["fingertip_crossed"])
        )

    def draw_debug_overlay(self, frame, result, target_non_hand_motion_threshold=300):
        annotated = frame.copy()
        frame_height, frame_width = annotated.shape[:2]

        if self.direction == "left_to_right":
            margin_x = self.partition_x + self.fingertip_margin
            target_label_anchor = (min(self.partition_x + 20, frame_width - 220), 32)
        else:
            margin_x = self.partition_x - self.fingertip_margin
            target_label_anchor = (20, 32)

        cv2.line(annotated, (self.partition_x, 0), (self.partition_x, frame_height - 1), (0, 0, 255), 2)
        cv2.line(
            annotated,
            (int(np.clip(margin_x, 0, frame_width - 1)), 0),
            (int(np.clip(margin_x, 0, frame_width - 1)), frame_height - 1),
            (255, 180, 0),
            2,
        )
        cv2.putText(
            annotated,
            f"Target side: {'right' if self.direction == 'left_to_right' else 'left'}",
            target_label_anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 220, 120),
            2,
            cv2.LINE_AA,
        )

        if result["hand_region_mask"] is not None:
            contours, _ = cv2.findContours(result["hand_region_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, (255, 0, 255), 2)

        if result["landmarks"] is not None:
            for landmark_index, point in enumerate(result["landmarks"]):
                color = (180, 180, 180)
                radius = 3
                if landmark_index == WRIST_INDEX:
                    color = (255, 255, 0)
                    radius = 5
                cv2.circle(annotated, point, radius, color, -1)

        for fingertip_name, fingertip_position in result["fingertip_positions"].items():
            is_crossed = fingertip_name in result["crossed_fingertips"]
            color = (0, 255, 0) if is_crossed else (0, 200, 255)
            cv2.circle(annotated, fingertip_position, 7, color, -1)
            cv2.putText(
                annotated,
                fingertip_name,
                (fingertip_position[0] + 6, max(18, fingertip_position[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        possible_block_without_fingertip = self.detect_block_without_fingertip_crossing(
            result,
            threshold=target_non_hand_motion_threshold,
        )

        info_lines = [
            f"Hand detected: {'YES' if result['hand_detected'] else 'NO'}",
            f"Fingertip crossed: {'YES' if result['fingertip_crossed'] else 'NO'}",
            f"Target non-hand area: {result['target_non_hand_motion_area']}",
            f"Possible block without fingertip: {'YES' if possible_block_without_fingertip else 'NO'}",
        ]
        y = 65
        for line in info_lines:
            cv2.putText(annotated, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(annotated, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            y += 30

        return annotated

    def close(self):
        self._hands.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optional MediaPipe-based hand and fingertip confirmation demo."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--partition-x", required=True, type=int, help="Partition x-position in the frame.")
    parser.add_argument("--direction", default="left_to_right", help="Expected crossing direction.")
    parser.add_argument("--fingertip-margin", type=int, default=10, help="Margin past the partition for fingertip crossing.")
    parser.add_argument("--hand-mask-padding", type=int, default=20, help="Extra padding around the landmark region.")
    parser.add_argument("--hand-mask-dilation", type=int, default=15, help="Dilation size for the approximate hand mask.")
    parser.add_argument(
        "--fingertips",
        default="thumb,index,middle",
        help="Comma-separated fingertips to monitor. Example: thumb,index,middle",
    )
    parser.add_argument(
        "--require-all-selected-fingertips",
        action="store_true",
        help="Require all selected fingertips to cross instead of just one.",
    )
    parser.add_argument(
        "--target-non-hand-motion-threshold",
        type=int,
        default=300,
        help="Threshold for flagging possible block motion without fingertip crossing.",
    )
    parser.add_argument("--resize-width", type=int, default=None, help="Optional output frame width.")
    parser.add_argument("--resize-height", type=int, default=None, help="Optional output frame height.")
    parser.add_argument(
        "--model-asset-path",
        default=None,
        help="Optional MediaPipe Hand Landmarker .task model path for Tasks-only installs.",
    )
    parser.add_argument("--display", action="store_true", help="Show debug windows while processing.")
    parser.add_argument("--save-output", default=None, help="Optional path for a saved debug video.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Could not find video: {video_path}")

    selected_fingertips = tuple(name.strip() for name in args.fingertips.split(",") if name.strip())
    confirmator = HandMotionConfirmator(
        partition_x=args.partition_x,
        direction=args.direction,
        fingertip_margin=args.fingertip_margin,
        hand_mask_padding=args.hand_mask_padding,
        hand_mask_dilation=args.hand_mask_dilation,
        selected_fingertips=selected_fingertips,
        require_all_selected_fingertips=args.require_all_selected_fingertips,
        model_asset_path=args.model_asset_path,
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=False,
    )

    writer = None
    fps = float(capture.get(cv2.CAP_PROP_FPS))
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
                resize_width=args.resize_width,
                resize_height=args.resize_height,
            )
            raw_motion_mask = background_subtractor.apply(resized_frame)
            cleaned_motion_mask = clean_motion_mask(raw_motion_mask)
            result = confirmator.analyze_frame(resized_frame, cleaned_motion_mask)
            overlay = confirmator.draw_debug_overlay(
                resized_frame,
                result,
                target_non_hand_motion_threshold=args.target_non_hand_motion_threshold,
            )

            if result["hand_detected"]:
                frames_with_hand_detected += 1
            if result["fingertip_crossed"]:
                frames_with_fingertip_crossing += 1
            if confirmator.detect_block_without_fingertip_crossing(
                result,
                threshold=args.target_non_hand_motion_threshold,
            ):
                frames_flagged_possible_block_without_fingertip += 1

            if args.save_output and writer is None:
                output_path = Path(args.save_output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter.fourcc(*"mp4v"),
                    fps if fps > 0 else 30.0,
                    (overlay.shape[1], overlay.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open output video for writing: {output_path}")

            if writer is not None:
                writer.write(overlay)

            if args.display:
                cv2.imshow("Hand Confirmation Overlay", overlay)
                cv2.imshow("Motion Mask", cleaned_motion_mask)

                hand_region_mask = result["hand_region_mask"]
                hand_motion_mask = result["hand_motion_mask"]
                non_hand_motion_mask = result["non_hand_motion_mask"]

                cv2.imshow(
                    "Hand Region Mask",
                    hand_region_mask if hand_region_mask is not None else np.zeros_like(cleaned_motion_mask),
                )
                cv2.imshow(
                    "Hand Motion Mask",
                    hand_motion_mask if hand_motion_mask is not None else np.zeros_like(cleaned_motion_mask),
                )
                cv2.imshow(
                    "Non-Hand Motion Mask",
                    non_hand_motion_mask if non_hand_motion_mask is not None else np.zeros_like(cleaned_motion_mask),
                )

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
