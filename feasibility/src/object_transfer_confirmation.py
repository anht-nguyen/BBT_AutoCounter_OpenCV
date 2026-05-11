from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ObjectTransferResult:
    target_motion_area: int
    target_motion_detected: bool
    persistence_detected: bool
    persistence_count: int
    absence_count: int
    target_mask: np.ndarray
    object_confirmed: bool


class ObjectTransferConfirmator:
    """Approximate object-transfer confirmation using non-hand motion on the target side.

    This helper does not detect a block directly. Instead, it looks for non-hand motion
    and short-lived persistence on the target side after a crossing event.
    """

    def __init__(
        self,
        partition_x: int,
        direction: str = "left_to_right",
        target_motion_threshold: int = 300,
        persistence_motion_threshold: int = 120,
        persistence_frames_required: int = 2,
        absence_reset_frames: int = 4,
    ) -> None:
        self.partition_x = int(partition_x)
        self.direction = str(direction)
        self.target_motion_threshold = max(int(target_motion_threshold), 0)
        self.persistence_motion_threshold = max(int(persistence_motion_threshold), 0)
        self.persistence_frames_required = max(int(persistence_frames_required), 1)
        self.absence_reset_frames = max(int(absence_reset_frames), 1)
        self._validate_inputs()
        self.reset()

    def _validate_inputs(self) -> None:
        if self.direction not in {"left_to_right", "right_to_left"}:
            raise ValueError("direction must be 'left_to_right' or 'right_to_left'")

    def reset(self) -> None:
        self.persistence_count = 0
        self.absence_count = 0

    def _target_side_mask(self, mask: np.ndarray) -> np.ndarray:
        frame_height, frame_width = mask.shape[:2]
        target_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        partition_x = int(np.clip(self.partition_x, 0, frame_width))

        if self.direction == "left_to_right":
            target_mask[:, partition_x:] = 255
        else:
            target_mask[:, :partition_x] = 255
        return target_mask

    def analyze_frame(self, non_hand_motion_mask: np.ndarray) -> ObjectTransferResult:
        target_mask = self._target_side_mask(non_hand_motion_mask)
        target_side_non_hand_motion = cv2.bitwise_and(non_hand_motion_mask, target_mask)
        target_motion_area = int(cv2.countNonZero(target_side_non_hand_motion))

        target_motion_detected = target_motion_area >= self.target_motion_threshold
        persistence_motion_seen = target_motion_area >= self.persistence_motion_threshold

        if persistence_motion_seen:
            self.persistence_count += 1
            self.absence_count = 0
        else:
            self.absence_count += 1
            if self.absence_count >= self.absence_reset_frames:
                self.persistence_count = 0

        persistence_detected = self.persistence_count >= self.persistence_frames_required
        object_confirmed = target_motion_detected or persistence_detected

        return ObjectTransferResult(
            target_motion_area=target_motion_area,
            target_motion_detected=target_motion_detected,
            persistence_detected=persistence_detected,
            persistence_count=self.persistence_count,
            absence_count=self.absence_count,
            target_mask=target_mask,
            object_confirmed=object_confirmed,
        )

    def draw_debug_overlay(self, frame: np.ndarray, result: ObjectTransferResult) -> np.ndarray:
        annotated = frame.copy()
        overlay = frame.copy()
        overlay[result.target_mask > 0] = (60, 120, 220)
        annotated = cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0)

        info_lines = [
            f"Target non-hand motion: {result.target_motion_area}",
            f"Target motion detected: {'YES' if result.target_motion_detected else 'NO'}",
            f"Persistence detected: {'YES' if result.persistence_detected else 'NO'}",
            f"Persistence frames: {result.persistence_count}",
            f"Object confirmed: {'YES' if result.object_confirmed else 'NO'}",
        ]
        y = 32
        for line in info_lines:
            cv2.putText(annotated, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(annotated, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y += 28

        return annotated
