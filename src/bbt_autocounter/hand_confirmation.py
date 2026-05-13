from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from .ui import draw_text_block


FINGERTIP_NAME_TO_INDEX = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
WRIST_INDEX = 0
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CANDIDATE_PATHS = (
    PROJECT_ROOT / "feasibility" / "models" / "hand_landmarker.task",
    PROJECT_ROOT / "models" / "hand_landmarker.task",
    PROJECT_ROOT / "feasibility" / "models" / "mediapipe" / "hand_landmarker.task",
)


@dataclass
class HandFrameResult:
    hand_detected: bool
    landmarks: list[tuple[int, int]] | None
    handedness: str | None
    fingertip_crossed: bool
    crossed_fingertips: list[str]
    fingertip_positions: dict[str, tuple[int, int]]
    hand_region_mask: np.ndarray | None
    hand_motion_mask: np.ndarray | None
    non_hand_motion_mask: np.ndarray
    target_non_hand_motion_area: int
    direction: str
    partition_x: int
    fingertip_margin: int

    def __getitem__(self, key: str):
        return getattr(self, key)


@dataclass
class LandmarkDetectionResult:
    hand_detected: bool
    landmarks: list[tuple[int, int]] | None
    handedness: str | None


class HandMotionConfirmator:
    def __init__(
        self,
        partition_x: int,
        direction: str = "left_to_right",
        fingertip_margin: int = 10,
        hand_mask_padding: int = 20,
        hand_mask_dilation: int = 15,
        selected_fingertips: tuple[str, ...] = ("thumb", "index", "middle"),
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
        for fingertip_name in self.selected_fingertips:
            if fingertip_name not in FINGERTIP_NAME_TO_INDEX:
                raise ValueError(f"Unsupported fingertip '{fingertip_name}'. Choose from: {', '.join(FINGERTIP_NAME_TO_INDEX)}")

    @staticmethod
    def _load_mediapipe() -> Any:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError("MediaPipe is required for hand confirmation. Install with: pip install mediapipe") from exc
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
                    "Provide a Hand Landmarker model with model_asset_path or --model-asset-path.\n"
                    f"Searched default paths:\n{searched_paths}"
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
            return mp, vision.HandLandmarker.create_from_options(options), "tasks"
        raise RuntimeError("Unsupported MediaPipe installation: no supported hand detection API found.")

    def detect_landmarks(self, frame: np.ndarray) -> LandmarkDetectionResult:
        frame_height, frame_width = frame.shape[:2]
        if self._backend == "solutions":
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._hands.process(frame_rgb)
            if not results.multi_hand_landmarks:
                return LandmarkDetectionResult(hand_detected=False, landmarks=None, handedness=None)
            hand_landmarks = results.multi_hand_landmarks[0]
            pixel_landmarks: list[tuple[int, int]] = []
            for landmark in hand_landmarks.landmark:
                x = int(np.clip(round(landmark.x * frame_width), 0, frame_width - 1))
                y = int(np.clip(round(landmark.y * frame_height), 0, frame_height - 1))
                pixel_landmarks.append((x, y))
            handedness = None if not results.multi_handedness else results.multi_handedness[0].classification[0].label
            return LandmarkDetectionResult(hand_detected=True, landmarks=pixel_landmarks, handedness=handedness)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = self._hands.detect(mp_image)
        if not results.hand_landmarks:
            return LandmarkDetectionResult(hand_detected=False, landmarks=None, handedness=None)
        pixel_landmarks: list[tuple[int, int]] = []
        for landmark in results.hand_landmarks[0]:
            x = int(np.clip(round(landmark.x * frame_width), 0, frame_width - 1))
            y = int(np.clip(round(landmark.y * frame_height), 0, frame_height - 1))
            pixel_landmarks.append((x, y))
        handedness = None if not getattr(results, "handedness", None) else results.handedness[0][0].category_name
        return LandmarkDetectionResult(hand_detected=True, landmarks=pixel_landmarks, handedness=handedness)

    def create_hand_region_mask(self, frame_shape: tuple[int, int, int], landmarks: list[tuple[int, int]] | None) -> np.ndarray:
        frame_height, frame_width = frame_shape[:2]
        hand_region_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        if not landmarks:
            return hand_region_mask
        points = np.asarray(landmarks, dtype=np.int32)
        cv2.fillConvexPoly(hand_region_mask, cv2.convexHull(points), 255)
        if self.hand_mask_padding > 0:
            x, y, w, h = cv2.boundingRect(points)
            x1 = max(0, x - self.hand_mask_padding)
            y1 = max(0, y - self.hand_mask_padding)
            x2 = min(frame_width, x + w + self.hand_mask_padding)
            y2 = min(frame_height, y + h + self.hand_mask_padding)
            hand_region_mask[y1:y2, x1:x2] = 255
        if self.hand_mask_dilation > 0:
            kernel = np.ones((max(1, self.hand_mask_dilation), max(1, self.hand_mask_dilation)), dtype=np.uint8)
            hand_region_mask = cv2.dilate(hand_region_mask, kernel, iterations=1)
        return hand_region_mask

    def split_motion_mask(self, motion_mask: np.ndarray, hand_region_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hand_motion_mask = cv2.bitwise_and(motion_mask, hand_region_mask)
        inverse_hand_mask = cv2.bitwise_not(hand_region_mask)
        non_hand_motion_mask = cv2.bitwise_and(motion_mask, inverse_hand_mask)
        return hand_motion_mask, non_hand_motion_mask

    def check_fingertip_crossing(self, landmarks: list[tuple[int, int]] | None) -> tuple[bool, list[str], dict[str, tuple[int, int]]]:
        if not landmarks:
            return False, [], {}
        crossed_fingertips: list[str] = []
        fingertip_positions: dict[str, tuple[int, int]] = {}
        for fingertip_name in self.selected_fingertips:
            fingertip_x, fingertip_y = landmarks[FINGERTIP_NAME_TO_INDEX[fingertip_name]]
            fingertip_positions[fingertip_name] = (fingertip_x, fingertip_y)
            crossed = fingertip_x > (self.partition_x + self.fingertip_margin) if self.direction == "left_to_right" else fingertip_x < (self.partition_x - self.fingertip_margin)
            if crossed:
                crossed_fingertips.append(fingertip_name)
        if self.require_all_selected_fingertips:
            selected_visible = [name for name in self.selected_fingertips if name in fingertip_positions]
            fingertip_crossed = bool(selected_visible) and len(crossed_fingertips) == len(selected_visible)
        else:
            fingertip_crossed = len(crossed_fingertips) > 0
        return fingertip_crossed, crossed_fingertips, fingertip_positions

    def compute_target_side_non_hand_motion_area(self, non_hand_motion_mask: np.ndarray) -> int:
        frame_width = non_hand_motion_mask.shape[1]
        if self.direction == "left_to_right":
            target_mask = non_hand_motion_mask[:, min(self.partition_x, frame_width):]
        else:
            target_mask = non_hand_motion_mask[:, :max(self.partition_x, 0)]
        return int(cv2.countNonZero(target_mask))

    def analyze_frame(self, frame: np.ndarray, motion_mask: np.ndarray) -> HandFrameResult:
        landmark_result = self.detect_landmarks(frame)
        hand_region_mask = None
        hand_motion_mask = None
        non_hand_motion_mask = motion_mask.copy()
        fingertip_crossed = False
        crossed_fingertips: list[str] = []
        fingertip_positions: dict[str, tuple[int, int]] = {}

        if landmark_result.hand_detected:
            hand_region_mask = self.create_hand_region_mask(frame.shape, landmark_result.landmarks)
            hand_motion_mask, non_hand_motion_mask = self.split_motion_mask(motion_mask, hand_region_mask)
            fingertip_crossed, crossed_fingertips, fingertip_positions = self.check_fingertip_crossing(landmark_result.landmarks)

        return HandFrameResult(
            hand_detected=landmark_result.hand_detected,
            landmarks=landmark_result.landmarks,
            handedness=landmark_result.handedness,
            fingertip_crossed=fingertip_crossed,
            crossed_fingertips=crossed_fingertips,
            fingertip_positions=fingertip_positions,
            hand_region_mask=hand_region_mask,
            hand_motion_mask=hand_motion_mask,
            non_hand_motion_mask=non_hand_motion_mask,
            target_non_hand_motion_area=self.compute_target_side_non_hand_motion_area(non_hand_motion_mask),
            direction=self.direction,
            partition_x=self.partition_x,
            fingertip_margin=self.fingertip_margin,
        )

    @staticmethod
    def detect_block_without_fingertip_crossing(result: HandFrameResult, threshold: int) -> bool:
        return int(result.target_non_hand_motion_area) >= int(threshold) and not result.fingertip_crossed

    def draw_debug_overlay(self, frame: np.ndarray, result: HandFrameResult, target_non_hand_motion_threshold: int = 300) -> np.ndarray:
        annotated = frame.copy()
        frame_height, frame_width = annotated.shape[:2]
        margin_x = self.partition_x + self.fingertip_margin if self.direction == "left_to_right" else self.partition_x - self.fingertip_margin

        cv2.line(annotated, (self.partition_x, 0), (self.partition_x, frame_height - 1), (0, 0, 255), 2)
        cv2.line(annotated, (int(np.clip(margin_x, 0, frame_width - 1)), 0), (int(np.clip(margin_x, 0, frame_width - 1)), frame_height - 1), (255, 180, 0), 2)
        if result.hand_region_mask is not None:
            contours, _ = cv2.findContours(result.hand_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, (255, 0, 255), 2)
        if result.landmarks is not None:
            for landmark_index, point in enumerate(result.landmarks):
                color = (255, 255, 0) if landmark_index == WRIST_INDEX else (180, 180, 180)
                radius = 5 if landmark_index == WRIST_INDEX else 3
                cv2.circle(annotated, point, radius, color, -1)
        for fingertip_name, fingertip_position in result.fingertip_positions.items():
            color = (0, 255, 0) if fingertip_name in result.crossed_fingertips else (0, 200, 255)
            cv2.circle(annotated, fingertip_position, 7, color, -1)
            cv2.putText(annotated, fingertip_name, (fingertip_position[0] + 6, max(18, fingertip_position[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        draw_text_block(
            annotated,
            [
                f"Hand detected: {'YES' if result.hand_detected else 'NO'}",
                f"Fingertip crossed: {'YES' if result.fingertip_crossed else 'NO'}",
                f"Target non-hand area: {result.target_non_hand_motion_area}",
                f"Possible block without fingertip: {'YES' if self.detect_block_without_fingertip_crossing(result, target_non_hand_motion_threshold) else 'NO'}",
            ],
            origin=(20, 65),
            font_scale=0.72,
            line_spacing=30,
        )
        return annotated

    def close(self) -> None:
        self._hands.close()
