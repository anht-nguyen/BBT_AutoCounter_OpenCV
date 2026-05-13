from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from motion_mask_cleaning import clean_motion_mask
except ImportError:
    try:
        from feasibility.src.motion_mask_cleaning import clean_motion_mask
    except ImportError:
        def clean_motion_mask(
            raw_mask,
            blur_kernel_size=5,
            threshold_value=200,
            morph_kernel_size=5,
            opening_iterations=1,
            closing_iterations=1,
        ):
            """Fallback cleaner used only if the shared module is not available."""
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


WAITING_FOR_START = "WAITING_FOR_START"
ARMED = "ARMED"
CANDIDATE_PENDING = "CANDIDATE_PENDING"
COOLDOWN = "COOLDOWN"


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
        self.current_side = "none"
        self.current_blob_center = None
        self.current_blob_area = 0.0
        self.current_left_edge = None
        self.current_right_edge = None

        # Hybrid mode keeps explicit state so we can wait for confirmation.
        self.state = WAITING_FOR_START
        self.candidate_pending = False
        self.candidate_frame_idx = None

        # Summary values help compare counting modes on the same pilot videos.
        self.total_candidate_crossings = 0
        self.confirmed_crossings = 0
        self.rejected_candidates = 0

    def classify_side(self, center_x):
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

    def is_valid_transition(self, previous_side, current_side):
        if previous_side is None or previous_side == current_side:
            return False
        if self.direction == "left_to_right":
            return previous_side == "left" and current_side == "right"
        if self.direction == "right_to_left":
            return previous_side == "right" and current_side == "left"
        raise ValueError("direction must be 'left_to_right' or 'right_to_left'")

    def _get_start_side(self):
        if self.direction == "left_to_right":
            return "left"
        if self.direction == "right_to_left":
            return "right"
        raise ValueError("direction must be 'left_to_right' or 'right_to_left'")

    def _get_target_side(self):
        return "right" if self._get_start_side() == "left" else "left"

    def _blob_majority_side(self, blob_info):
        if blob_info is None:
            return "none"

        left_span = max(0, self.partition_x - blob_info["left_edge"])
        right_span = max(0, blob_info["right_edge"] - self.partition_x)
        if left_span > right_span:
            return "left"
        if right_span > left_span:
            return "right"
        return "neutral"

    def _blob_is_on_start_side(self, blob_info, current_side):
        if blob_info is None:
            return False

        start_side = self._get_start_side()
        majority_side = self._blob_majority_side(blob_info)
        return current_side == start_side or majority_side == start_side

    def _cooldown_remaining(self, frame_idx):
        return max(0, self.cooldown_frames - (int(frame_idx) - self.last_count_frame))

    def _candidate_age_frames(self, frame_idx):
        if self.candidate_frame_idx is None:
            return None
        return int(frame_idx) - int(self.candidate_frame_idx)

    def has_leading_edge_crossed(self, blob_info):
        if blob_info is None:
            return False
        if self.direction == "left_to_right":
            return blob_info["right_edge"] > (self.partition_x + self.leading_edge_margin)
        if self.direction == "right_to_left":
            return blob_info["left_edge"] < (self.partition_x - self.leading_edge_margin)
        raise ValueError("direction must be 'left_to_right' or 'right_to_left'")

    def compute_target_side_motion_area(self, cleaned_mask):
        """Measure how much motion is visible on the target side of the partition."""
        frame_width = cleaned_mask.shape[1]
        if self.direction == "left_to_right":
            target_mask = cleaned_mask[:, min(self.partition_x, frame_width):]
        elif self.direction == "right_to_left":
            target_mask = cleaned_mask[:, :max(self.partition_x, 0)]
        else:
            raise ValueError("direction must be 'left_to_right' or 'right_to_left'")

        # Counting white pixels is a simple first-pass motion measure.
        return int(cv2.countNonZero(target_mask))

    def has_target_confirmation(self, cleaned_mask):
        target_motion_area = self.compute_target_side_motion_area(cleaned_mask)
        target_confirmed = target_motion_area >= self.target_motion_area_threshold
        return target_confirmed, target_motion_area

    def find_main_blob(self, cleaned_mask):
        _, frame_width = cleaned_mask.shape[:2]
        x1 = self.partition_x - (self.crossing_zone_width // 2)
        x2 = self.partition_x + (self.crossing_zone_width // 2)
        x1 = max(0, x1)
        x2 = min(frame_width, x2)
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

    def _arm_if_start_side_seen(self, blob_info, current_side):
        if self._blob_is_on_start_side(blob_info, current_side):
            self.armed_for_crossing = True
            if self.confirmation_mode == "hybrid" and self.state == WAITING_FOR_START:
                self.state = ARMED

    def _confirm_count(self, frame_idx):
        self.count += 1
        self.confirmed_crossings += 1
        self.last_count_frame = int(frame_idx)
        self.last_event_detected = True
        self.armed_for_crossing = False
        self.candidate_pending = False
        self.candidate_frame_idx = None
        self.state = COOLDOWN

    def _make_result(
        self,
        frame_idx,
        event_detected,
        score_status,
        blob_info,
        current_side,
        target_motion_area,
        target_confirmed,
    ):
        cooldown_remaining = self._cooldown_remaining(frame_idx)
        candidate_age_frames = self._candidate_age_frames(frame_idx)
        return {
            "count": self.count,
            "event_detected": event_detected,
            "motion_scored": event_detected,
            "candidate_pending": self.candidate_pending,
            "candidate_frame_idx": self.candidate_frame_idx,
            "candidate_age_frames": candidate_age_frames,
            "state": self.state,
            "current_side": current_side,
            "blob_center": self.current_blob_center,
            "blob_area": self.current_blob_area,
            "left_edge": self.current_left_edge,
            "right_edge": self.current_right_edge,
            "bbox": None if blob_info is None else blob_info["bbox"],
            "zone_bounds": None if blob_info is None else blob_info["zone_bounds"],
            "target_motion_area": target_motion_area,
            "target_confirmed": target_confirmed,
            "cooldown_remaining": cooldown_remaining,
            "confirmation_mode": self.confirmation_mode,
            "crossing_method": self.confirmation_mode,
            "leading_edge_margin": self.leading_edge_margin,
            "armed_for_crossing": self.armed_for_crossing,
            "score_status": score_status,
        }

    def update(self, cleaned_mask, frame_idx):
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
            self.current_blob_area = blob_info["area"]
            self.current_left_edge = blob_info["left_edge"]
            self.current_right_edge = blob_info["right_edge"]
            current_side = self.classify_side(blob_info["center"][0])

        previous_side_before_update = self.previous_side
        if blob_info is not None and current_side not in ("none", "neutral"):
            self.previous_side = current_side

        if self.confirmation_mode == "center":
            cooldown_remaining = self._cooldown_remaining(frame_idx)
            if cooldown_remaining > 0:
                self.state = COOLDOWN
                score_status = "cooldown active"
            else:
                self._arm_if_start_side_seen(blob_info, current_side)
                self.state = ARMED if self.armed_for_crossing else WAITING_FOR_START
                if (
                    blob_info is not None
                    and current_side not in ("none", "neutral")
                    and self.is_valid_transition(previous_side_before_update, current_side)
                    and self.armed_for_crossing
                ):
                    self._confirm_count(frame_idx)
                    event_detected = True
                    score_status = "center crossing confirmed"
                elif blob_info is not None:
                    score_status = "waiting for center crossing"

        elif self.confirmation_mode == "leading_edge":
            cooldown_remaining = self._cooldown_remaining(frame_idx)
            if cooldown_remaining > 0:
                self.state = COOLDOWN
                score_status = "cooldown active"
            else:
                self._arm_if_start_side_seen(blob_info, current_side)
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
            cooldown_remaining = self._cooldown_remaining(frame_idx)

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

        if blob_info is None and self.confirmation_mode != "hybrid":
            score_status = "no blob"
        elif blob_info is None and self.state == WAITING_FOR_START:
            score_status = "no blob"

        return self._make_result(
            frame_idx=frame_idx,
            event_detected=event_detected,
            score_status=score_status,
            blob_info=blob_info,
            current_side=current_side,
            target_motion_area=target_motion_area,
            target_confirmed=target_confirmed,
        )

    def draw_debug_overlay(self, frame, result):
        annotated = frame.copy()
        frame_height, frame_width = annotated.shape[:2]

        x1 = max(0, self.partition_x - (self.crossing_zone_width // 2))
        x2 = min(frame_width, self.partition_x + (self.crossing_zone_width // 2))
        left_dead_zone = int(round(self.partition_x - (self.dead_zone_width / 2.0)))
        right_dead_zone = int(round(self.partition_x + (self.dead_zone_width / 2.0)))
        left_margin_line = int(round(self.partition_x - self.leading_edge_margin))
        right_margin_line = int(round(self.partition_x + self.leading_edge_margin))

        # The partition line is the main reference for all crossing decisions.
        cv2.line(annotated, (self.partition_x, 0), (self.partition_x, frame_height - 1), (0, 0, 255), 2)
        cv2.rectangle(annotated, (x1, 0), (x2, frame_height - 1), (255, 0, 0), 2)
        cv2.line(annotated, (max(left_dead_zone, 0), 0), (max(left_dead_zone, 0), frame_height - 1), (0, 255, 255), 2)
        cv2.line(
            annotated,
            (min(right_dead_zone, frame_width - 1), 0),
            (min(right_dead_zone, frame_width - 1), frame_height - 1),
            (0, 255, 255),
            2,
        )
        cv2.line(annotated, (max(left_margin_line, 0), 0), (max(left_margin_line, 0), frame_height - 1), (255, 180, 0), 2)
        cv2.line(
            annotated,
            (min(right_margin_line, frame_width - 1), 0),
            (min(right_margin_line, frame_width - 1), frame_height - 1),
            (255, 180, 0),
            2,
        )

        target_side_label = f"Target side: {self._get_target_side()}"
        cv2.putText(annotated, target_side_label, (max(x2 + 10, 20), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 120), 2, cv2.LINE_AA)

        if result["bbox"] is not None:
            x, y, w, h = result["bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (50, 220, 50), 2)

        if result["blob_center"] is not None:
            cx, cy = result["blob_center"]
            cv2.circle(annotated, (cx, cy), 5, (50, 220, 50), -1)

        if result["left_edge"] is not None:
            cv2.line(annotated, (int(result["left_edge"]), 0), (int(result["left_edge"]), frame_height - 1), (120, 255, 120), 1)
        if result["right_edge"] is not None:
            cv2.line(annotated, (int(result["right_edge"]), 0), (int(result["right_edge"]), frame_height - 1), (120, 255, 120), 1)

        candidate_age = result["candidate_age_frames"]
        info_lines = [
            f"Count: {result['count']}",
            f"Mode: {result['confirmation_mode']}",
            f"State: {result['state']}",
            f"Current side: {result['current_side']}",
            f"Candidate pending: {'YES' if result['candidate_pending'] else 'NO'}",
            f"Candidate age: {candidate_age if candidate_age is not None else '-'}",
            f"Target motion area: {result['target_motion_area']}",
            f"Target confirmed: {'YES' if result['target_confirmed'] else 'NO'}",
            f"Cooldown: {result['cooldown_remaining']}",
        ]
        y = 32
        for line in info_lines:
            cv2.putText(annotated, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(annotated, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            y += 28

        if result["candidate_pending"]:
            cv2.putText(
                annotated,
                "CANDIDATE",
                (20, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 220, 255),
                3,
                cv2.LINE_AA,
            )
        if result["event_detected"]:
            cv2.putText(
                annotated,
                "CROSSING CONFIRMED!",
                (20, y + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

        return annotated


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect transfer-like crossings based on a cleaned motion mask."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--partition-x", required=True, type=int, help="Partition x-position in the frame.")
    parser.add_argument("--direction", default="left_to_right", help="Expected crossing direction.")
    parser.add_argument("--crossing-zone-width", type=int, default=100, help="Width of the detection zone.")
    parser.add_argument("--dead-zone-width", type=int, default=20, help="Neutral zone width around the partition.")
    parser.add_argument("--min-area", type=int, default=500, help="Minimum blob area to keep.")
    parser.add_argument("--cooldown-frames", type=int, default=20, help="Frames to wait before counting again.")
    parser.add_argument(
        "--confirmation-mode",
        choices=["center", "leading_edge", "hybrid"],
        default="hybrid",
        help="How strict the detector should be before counting.",
    )
    parser.add_argument("--leading-edge-margin", type=int, default=10, help="Margin past the partition for edge crossing.")
    parser.add_argument(
        "--target-confirmation-window-frames",
        type=int,
        default=10,
        help="How many frames hybrid mode waits for target-side confirmation.",
    )
    parser.add_argument(
        "--target-motion-area-threshold",
        type=int,
        default=300,
        help="Minimum target-side motion needed to confirm a hybrid candidate.",
    )
    parser.add_argument("--resize-width", type=int, default=None, help="Optional output frame width.")
    parser.add_argument("--resize-height", type=int, default=None, help="Optional output frame height.")
    parser.add_argument("--display", action="store_true", help="Show debug windows while processing.")
    parser.add_argument("--save-output", default=None, help="Optional path for a saved debug video.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Could not find video: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    counter = CrossingCounter(
        partition_x=args.partition_x,
        direction=args.direction,
        crossing_zone_width=args.crossing_zone_width,
        dead_zone_width=args.dead_zone_width,
        min_area=args.min_area,
        cooldown_frames=args.cooldown_frames,
        leading_edge_margin=args.leading_edge_margin,
        confirmation_mode=args.confirmation_mode,
        target_confirmation_window_frames=args.target_confirmation_window_frames,
        target_motion_area_threshold=args.target_motion_area_threshold,
    )
    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=False,
    )

    writer = None
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_index = 0

    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break

            frame_index += 1
            resized_frame = resize_frame(frame, resize_width=args.resize_width, resize_height=args.resize_height)
            raw_mask = background_subtractor.apply(resized_frame)
            cleaned_mask = clean_motion_mask(raw_mask)
            result = counter.update(cleaned_mask, frame_idx=frame_index)
            annotated_frame = counter.draw_debug_overlay(resized_frame, result)

            if args.save_output and writer is None:
                output_path = Path(args.save_output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter.fourcc(*"mp4v"),
                    fps if fps > 0 else 30.0,
                    (annotated_frame.shape[1], annotated_frame.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open output video for writing: {output_path}")

            if writer is not None:
                writer.write(annotated_frame)

            if args.display:
                cv2.imshow("Crossing Counter", annotated_frame)
                cv2.imshow("Cleaned Motion Mask", cleaned_mask)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    cv2.waitKey(0)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"Final automated count: {counter.count}")
    print(f"Confirmation mode: {counter.confirmation_mode}")
    print(f"Leading edge margin: {counter.leading_edge_margin}")
    print(f"Total candidate crossings: {counter.total_candidate_crossings}")
    print(f"Confirmed crossings: {counter.confirmed_crossings}")
    print(f"Rejected candidates: {counter.rejected_candidates}")
    print(f"Final count: {counter.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
