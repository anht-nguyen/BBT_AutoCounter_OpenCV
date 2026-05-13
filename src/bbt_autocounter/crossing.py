from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .ui import draw_text_block


WAITING_FOR_START = "WAITING_FOR_START"
ARMED = "ARMED"
CANDIDATE_PENDING = "CANDIDATE_PENDING"
COOLDOWN = "COOLDOWN"


@dataclass
class CrossingResult:
    count: int
    event_detected: bool
    motion_scored: bool
    candidate_pending: bool
    candidate_frame_idx: int | None
    candidate_age_frames: int | None
    state: str
    current_side: str
    blob_center: tuple[int, int] | None
    blob_area: float
    left_edge: int | None
    right_edge: int | None
    bbox: tuple[int, int, int, int] | None
    zone_bounds: tuple[int, int] | None
    target_motion_area: int
    target_confirmed: bool
    cooldown_remaining: int
    confirmation_mode: str
    crossing_method: str
    leading_edge_margin: int
    armed_for_crossing: bool
    score_status: str

    def __getitem__(self, key: str):
        return getattr(self, key)


class CrossingCounter:
    def __init__(
        self,
        partition_x: int,
        direction: str = "left_to_right",
        crossing_zone_width: int = 100,
        dead_zone_width: int = 20,
        min_area: int = 500,
        cooldown_frames: int = 20,
        leading_edge_margin: int = 10,
        confirmation_mode: str = "hybrid",
        target_confirmation_window_frames: int = 10,
        target_motion_area_threshold: int = 300,
    ) -> None:
        self.partition_x = int(partition_x)
        self.direction = str(direction)
        self.crossing_zone_width = max(int(crossing_zone_width), 1)
        self.dead_zone_width = max(int(dead_zone_width), 0)
        self.min_area = max(int(min_area), 1)
        self.cooldown_frames = max(int(cooldown_frames), 0)
        self.leading_edge_margin = max(int(leading_edge_margin), 0)
        self.confirmation_mode = str(confirmation_mode)
        self.target_confirmation_window_frames = max(int(target_confirmation_window_frames), 1)
        self.target_motion_area_threshold = max(int(target_motion_area_threshold), 0)
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.previous_side = None
        self.armed_for_crossing = False
        self.last_count_frame = -self.cooldown_frames
        self.last_event_detected = False
        self.current_blob_center = None
        self.current_blob_area = 0.0
        self.current_left_edge = None
        self.current_right_edge = None
        self.state = WAITING_FOR_START
        self.candidate_pending = False
        self.candidate_frame_idx = None
        self.total_candidate_crossings = 0
        self.confirmed_crossings = 0
        self.rejected_candidates = 0

    def classify_side(self, center_x: int | None) -> str:
        if center_x is None:
            return "none"
        half_dead_zone = self.dead_zone_width / 2.0
        left_dead_zone = self.partition_x - half_dead_zone
        right_dead_zone = self.partition_x + half_dead_zone
        if center_x < left_dead_zone:
            return "left"
        if center_x > right_dead_zone:
            return "right"
        return "neutral"

    def is_valid_transition(self, previous_side: str | None, current_side: str) -> bool:
        if previous_side is None or previous_side == current_side:
            return False
        if self.direction == "left_to_right":
            return previous_side == "left" and current_side == "right"
        if self.direction == "right_to_left":
            return previous_side == "right" and current_side == "left"
        raise ValueError("direction must be 'left_to_right' or 'right_to_left'")


    def _get_start_side(self) -> str:
        if self.direction == "left_to_right":
            return "left"
        if self.direction == "right_to_left":
            return "right"
        raise ValueError("direction must be 'left_to_right' or 'right_to_left'")

    def _get_target_side(self) -> str:
        return "right" if self._get_start_side() == "left" else "left"

    def _blob_majority_side(self, blob_info: dict[str, object] | None) -> str:
        if blob_info is None:
            return "none"
        left_span = max(0, self.partition_x - int(blob_info["left_edge"]))
        right_span = max(0, int(blob_info["right_edge"]) - self.partition_x)
        if left_span > right_span:
            return "left"
        if right_span > left_span:
            return "right"
        return "neutral"

    def _blob_is_on_start_side(self, blob_info: dict[str, object] | None, current_side: str) -> bool:
        if blob_info is None:
            return False
        return current_side == self._get_start_side() or self._blob_majority_side(blob_info) == self._get_start_side()

    def _cooldown_remaining(self, frame_idx: int) -> int:
        return max(0, self.cooldown_frames - (int(frame_idx) - self.last_count_frame))

    def _candidate_age_frames(self, frame_idx: int) -> int | None:
        if self.candidate_frame_idx is None:
            return None
        return int(frame_idx) - int(self.candidate_frame_idx)

    def has_leading_edge_crossed(self, blob_info: dict[str, object] | None) -> bool:
        if blob_info is None:
            return False
        if self.direction == "left_to_right":
            return int(blob_info["right_edge"]) > (self.partition_x + self.leading_edge_margin)
        if self.direction == "right_to_left":
            return int(blob_info["left_edge"]) < (self.partition_x - self.leading_edge_margin)
        raise ValueError("direction must be 'left_to_right' or 'right_to_left'")

    def compute_target_side_motion_area(self, cleaned_mask: np.ndarray) -> int:
        frame_width = cleaned_mask.shape[1]
        if self.direction == "left_to_right":
            target_mask = cleaned_mask[:, min(self.partition_x, frame_width):]
        elif self.direction == "right_to_left":
            target_mask = cleaned_mask[:, :max(self.partition_x, 0)]
        else:
            raise ValueError("direction must be 'left_to_right' or 'right_to_left'")
        return int(cv2.countNonZero(target_mask))

    def has_target_confirmation(self, cleaned_mask: np.ndarray) -> tuple[bool, int]:
        target_motion_area = self.compute_target_side_motion_area(cleaned_mask)
        return target_motion_area >= self.target_motion_area_threshold, target_motion_area

    def find_main_blob(self, cleaned_mask: np.ndarray) -> dict[str, object] | None:
        _, frame_width = cleaned_mask.shape[:2]
        x1 = max(0, self.partition_x - (self.crossing_zone_width // 2))
        x2 = min(frame_width, self.partition_x + (self.crossing_zone_width // 2))
        if x2 <= x1:
            return None
        zone_mask = cleaned_mask[:, x1:x2]
        contours, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [contour for contour in contours if cv2.contourArea(contour) >= self.min_area]
        if not valid_contours:
            return None
        contour = max(valid_contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if abs(float(moments["m00"])) > 1e-8:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx = x + (w // 2)
            cy = y + (h // 2)
        full_frame_x = x + x1
        return {
            "contour": contour,
            "area": area,
            "bbox": (full_frame_x, y, w, h),
            "center": (cx + x1, cy),
            "left_edge": full_frame_x,
            "right_edge": full_frame_x + w,
            "zone_bounds": (x1, x2),
        }

    def _confirm_count(self, frame_idx: int) -> None:
        self.count += 1
        self.confirmed_crossings += 1
        self.last_count_frame = int(frame_idx)
        self.last_event_detected = True
        self.armed_for_crossing = False
        self.candidate_pending = False
        self.candidate_frame_idx = None
        self.state = COOLDOWN

    def update(self, cleaned_mask: np.ndarray, frame_idx: int) -> CrossingResult:
        blob_info = self.find_main_blob(cleaned_mask)
        current_side = "none"
        target_confirmed, target_motion_area = self.has_target_confirmation(cleaned_mask)
        event_detected = False
        score_status = "no blob"
        self.last_event_detected = False

        if blob_info is None:
            self.current_blob_center = None
            self.current_blob_area = 0.0
            self.current_left_edge = None
            self.current_right_edge = None
        else:
            self.current_blob_center = blob_info["center"]
            self.current_blob_area = float(blob_info["area"])
            self.current_left_edge = int(blob_info["left_edge"])
            self.current_right_edge = int(blob_info["right_edge"])
            current_side = self.classify_side(self.current_blob_center[0])

        previous_side_before_update = self.previous_side
        if blob_info is not None and current_side not in ("none", "neutral"):
            self.previous_side = current_side

        cooldown_remaining = self._cooldown_remaining(frame_idx)
        if self.confirmation_mode == "center":
            if cooldown_remaining > 0:
                self.state = COOLDOWN
                score_status = "cooldown active"
            else:
                if self._blob_is_on_start_side(blob_info, current_side):
                    self.armed_for_crossing = True
                self.state = ARMED if self.armed_for_crossing else WAITING_FOR_START
                if blob_info is not None and current_side not in ("none", "neutral") and self.is_valid_transition(previous_side_before_update, current_side) and self.armed_for_crossing:
                    self._confirm_count(frame_idx)
                    event_detected = True
                    score_status = "center crossing confirmed"
                elif blob_info is not None:
                    score_status = "waiting for center crossing"
        elif self.confirmation_mode == "leading_edge":
            if cooldown_remaining > 0:
                self.state = COOLDOWN
                score_status = "cooldown active"
            else:
                if self._blob_is_on_start_side(blob_info, current_side):
                    self.armed_for_crossing = True
                self.state = ARMED if self.armed_for_crossing else WAITING_FOR_START
                if blob_info is not None and self.armed_for_crossing and self.has_leading_edge_crossed(blob_info):
                    self._confirm_count(frame_idx)
                    event_detected = True
                    score_status = "leading edge crossing confirmed"
                elif blob_info is not None and self.armed_for_crossing:
                    score_status = "armed, waiting for leading edge crossing"
                elif blob_info is not None:
                    score_status = "waiting for start side"
        elif self.confirmation_mode == "hybrid":
            if self.state == COOLDOWN:
                if cooldown_remaining > 0:
                    score_status = "cooldown active"
                else:
                    self.state = WAITING_FOR_START
                    score_status = "cooldown finished"
            if self.state == WAITING_FOR_START:
                if self._blob_is_on_start_side(blob_info, current_side):
                    self.armed_for_crossing = True
                    self.state = ARMED
                    score_status = "start side seen, detector armed"
                else:
                    self.armed_for_crossing = False
                    score_status = "waiting for start side"
            elif self.state == ARMED:
                self.armed_for_crossing = True
                if blob_info is not None and self.has_leading_edge_crossed(blob_info):
                    self.candidate_pending = True
                    self.candidate_frame_idx = int(frame_idx)
                    self.total_candidate_crossings += 1
                    self.state = CANDIDATE_PENDING
                    score_status = "candidate crossing detected"
                else:
                    score_status = "armed, waiting for leading edge crossing"
            elif self.state == CANDIDATE_PENDING:
                self.candidate_pending = True
                candidate_age_frames = self._candidate_age_frames(frame_idx) or 0
                if target_confirmed:
                    self._confirm_count(frame_idx)
                    event_detected = True
                    score_status = "hybrid crossing confirmed"
                elif candidate_age_frames > self.target_confirmation_window_frames:
                    self.candidate_pending = False
                    self.candidate_frame_idx = None
                    self.armed_for_crossing = False
                    self.rejected_candidates += 1
                    self.state = WAITING_FOR_START
                    score_status = "candidate rejected, no target confirmation"
                else:
                    score_status = "candidate pending target confirmation"
        else:
            raise ValueError("confirmation_mode must be 'center', 'leading_edge', or 'hybrid'")

        candidate_age_frames = self._candidate_age_frames(frame_idx)
        return CrossingResult(
            count=self.count,
            event_detected=event_detected,
            motion_scored=event_detected,
            candidate_pending=self.candidate_pending,
            candidate_frame_idx=self.candidate_frame_idx,
            candidate_age_frames=candidate_age_frames,
            state=self.state,
            current_side=current_side,
            blob_center=self.current_blob_center,
            blob_area=self.current_blob_area,
            left_edge=self.current_left_edge,
            right_edge=self.current_right_edge,
            bbox=None if blob_info is None else blob_info["bbox"],
            zone_bounds=None if blob_info is None else blob_info["zone_bounds"],
            target_motion_area=target_motion_area,
            target_confirmed=target_confirmed,
            cooldown_remaining=cooldown_remaining,
            confirmation_mode=self.confirmation_mode,
            crossing_method=self.confirmation_mode,
            leading_edge_margin=self.leading_edge_margin,
            armed_for_crossing=self.armed_for_crossing,
            score_status=score_status if blob_info is not None or self.state != WAITING_FOR_START else "no blob",
        )

    def draw_debug_overlay(self, frame: np.ndarray, result: CrossingResult) -> np.ndarray:
        annotated = frame.copy()
        frame_height, frame_width = annotated.shape[:2]
        x1 = max(0, self.partition_x - (self.crossing_zone_width // 2))
        x2 = min(frame_width, self.partition_x + (self.crossing_zone_width // 2))
        left_dead_zone = int(round(self.partition_x - (self.dead_zone_width / 2.0)))
        right_dead_zone = int(round(self.partition_x + (self.dead_zone_width / 2.0)))
        left_margin_line = int(round(self.partition_x - self.leading_edge_margin))
        right_margin_line = int(round(self.partition_x + self.leading_edge_margin))

        cv2.line(annotated, (self.partition_x, 0), (self.partition_x, frame_height - 1), (0, 0, 255), 2)
        cv2.rectangle(annotated, (x1, 0), (x2, frame_height - 1), (255, 0, 0), 2)
        cv2.line(annotated, (max(left_dead_zone, 0), 0), (max(left_dead_zone, 0), frame_height - 1), (0, 255, 255), 2)
        cv2.line(annotated, (min(right_dead_zone, frame_width - 1), 0), (min(right_dead_zone, frame_width - 1), frame_height - 1), (0, 255, 255), 2)
        cv2.line(annotated, (max(left_margin_line, 0), 0), (max(left_margin_line, 0), frame_height - 1), (255, 180, 0), 2)
        cv2.line(annotated, (min(right_margin_line, frame_width - 1), 0), (min(right_margin_line, frame_width - 1), frame_height - 1), (255, 180, 0), 2)

        if result.bbox is not None:
            x, y, w, h = result.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (50, 220, 50), 2)
        if result.blob_center is not None:
            cv2.circle(annotated, result.blob_center, 5, (50, 220, 50), -1)
        if result.left_edge is not None:
            cv2.line(annotated, (result.left_edge, 0), (result.left_edge, frame_height - 1), (120, 255, 120), 1)
        if result.right_edge is not None:
            cv2.line(annotated, (result.right_edge, 0), (result.right_edge, frame_height - 1), (120, 255, 120), 1)

        draw_text_block(
            annotated,
            [
                f"Count: {result.count}",
                f"Mode: {result.confirmation_mode}",
                f"State: {result.state}",
                f"Current side: {result.current_side}",
                f"Candidate pending: {'YES' if result.candidate_pending else 'NO'}",
                f"Candidate age: {result.candidate_age_frames if result.candidate_age_frames is not None else '-'}",
                f"Target motion area: {result.target_motion_area}",
                f"Target confirmed: {'YES' if result.target_confirmed else 'NO'}",
                f"Cooldown: {result.cooldown_remaining}",
            ],
        )
        if result.candidate_pending:
            draw_text_block(annotated, ["CANDIDATE"], origin=(20, 295), font_scale=1.0, color=(0, 220, 255), thickness=3)
        if result.event_detected:
            draw_text_block(annotated, ["CROSSING CONFIRMED!"], origin=(20, 340), font_scale=1.0, color=(0, 255, 0), thickness=3)
        return annotated
