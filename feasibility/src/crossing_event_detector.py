from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from mask_cleaning import clean_motion_mask
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
            """Simple fallback cleaner used if the shared module is not available."""
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
        crossing_method: str = "leading_edge",
    ) -> None:
        self.partition_x = int(partition_x)
        self.direction = str(direction)
        self.crossing_zone_width = max(int(crossing_zone_width), 1)
        self.dead_zone_width = max(int(dead_zone_width), 0)
        self.min_area = max(int(min_area), 1)
        self.cooldown_frames = max(int(cooldown_frames), 0)
        self.leading_edge_margin = max(int(leading_edge_margin), 0)
        self.crossing_method = str(crossing_method)
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.previous_side = None
        self.armed_for_crossing = False
        self.last_count_frame = -self.cooldown_frames
        self.last_event_detected = False
        self.last_scored_frame = -1
        self.current_side = "none"
        self.current_blob_center = None
        self.current_blob_area = 0.0
        self.current_left_edge = None
        self.current_right_edge = None

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
        if previous_side is None:
            return False
        if previous_side == current_side:
            return False

        if self.direction == "left_to_right":
            return previous_side == "left" and current_side == "right"
        if self.direction == "right_to_left":
            return previous_side == "right" and current_side == "left"
        raise ValueError("direction must be 'left_to_right' or 'right_to_left'")

    def has_leading_edge_crossed(self, blob_info):
        if blob_info is None:
            return False

        if self.direction == "left_to_right":
            return blob_info["right_edge"] > (self.partition_x + self.leading_edge_margin)
        if self.direction == "right_to_left":
            return blob_info["left_edge"] < (self.partition_x - self.leading_edge_margin)
        raise ValueError("direction must be 'left_to_right' or 'right_to_left'")

    def _get_start_side(self):
        if self.direction == "left_to_right":
            return "left"
        if self.direction == "right_to_left":
            return "right"
        raise ValueError("direction must be 'left_to_right' or 'right_to_left'")

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
        full_frame_bbox = (full_frame_x, y, w, h)
        full_frame_center = (cx + x1, cy)
        left_edge = full_frame_x
        right_edge = full_frame_x + w
        return {
            "contour": contour,
            "area": area,
            "bbox": full_frame_bbox,
            "center": full_frame_center,
            "left_edge": left_edge,
            "right_edge": right_edge,
            "zone_bounds": (x1, x2),
        }

    def update(self, cleaned_mask, frame_idx):
        blob_info = self.find_main_blob(cleaned_mask)
        event_detected = False
        zone_bounds = None
        bbox = None
        blob_center = None
        blob_area = 0.0
        left_edge = None
        right_edge = None
        current_side = "none"
        cooldown_remaining = self._cooldown_remaining(frame_idx)

        if blob_info is None:
            self.last_event_detected = False
            self.current_side = "none"
            self.current_blob_center = None
            self.current_blob_area = 0.0
            self.current_left_edge = None
            self.current_right_edge = None
            return {
                "count": self.count,
                "event_detected": False,
                "motion_scored": False,
                "score_status": "no blob",
                "crossing_method": self.crossing_method,
                "leading_edge_margin": self.leading_edge_margin,
                "current_side": self.current_side,
                "previous_side": self.previous_side,
                "armed_for_crossing": self.armed_for_crossing,
                "blob_center": self.current_blob_center,
                "blob_area": self.current_blob_area,
                "left_edge": self.current_left_edge,
                "right_edge": self.current_right_edge,
                "bbox": None,
                "zone_bounds": None,
                "cooldown_remaining": cooldown_remaining,
            }

        bbox = blob_info["bbox"]
        blob_center = blob_info["center"]
        blob_area = blob_info["area"]
        left_edge = blob_info["left_edge"]
        right_edge = blob_info["right_edge"]
        zone_bounds = blob_info["zone_bounds"]
        current_side = self.classify_side(blob_center[0])
        observed_previous_side = self.previous_side
        is_in_cooldown = cooldown_remaining > 0

        if self._blob_is_on_start_side(blob_info, current_side):
            self.armed_for_crossing = True

        if self.crossing_method == "center":
            if current_side != "neutral" and current_side != "none":
                if self.is_valid_transition(observed_previous_side, current_side) and not is_in_cooldown:
                    self.count += 1
                    event_detected = True
                    self.last_count_frame = int(frame_idx)
                    self.last_scored_frame = int(frame_idx)
                    self.armed_for_crossing = False
                    cooldown_remaining = self._cooldown_remaining(frame_idx)
                self.previous_side = current_side
        elif self.crossing_method == "leading_edge":
            leading_edge_crossed = self.has_leading_edge_crossed(blob_info)
            if self.armed_for_crossing and leading_edge_crossed and not is_in_cooldown:
                self.count += 1
                event_detected = True
                self.last_count_frame = int(frame_idx)
                self.last_scored_frame = int(frame_idx)
                self.armed_for_crossing = False
                cooldown_remaining = self._cooldown_remaining(frame_idx)
            if current_side not in ("none", "neutral"):
                self.previous_side = current_side
        else:
            raise ValueError("crossing_method must be 'center' or 'leading_edge'")

        self.last_event_detected = event_detected
        self.current_side = current_side
        self.current_blob_center = blob_center
        self.current_blob_area = blob_area
        self.current_left_edge = left_edge
        self.current_right_edge = right_edge

        if event_detected:
            score_status = "scored"
        elif self.crossing_method == "leading_edge" and not self.armed_for_crossing:
            score_status = "waiting to re-arm on start side"
        elif current_side == "neutral":
            score_status = "inside dead zone"
        elif cooldown_remaining > 0:
            score_status = "motion seen, cooldown active"
        elif self.crossing_method == "leading_edge" and self.has_leading_edge_crossed(blob_info):
            score_status = "leading edge crossed, waiting for cooldown"
        elif self.crossing_method == "leading_edge" and self.armed_for_crossing:
            score_status = "armed, waiting for leading edge crossing"
        elif self.previous_side == current_side:
            score_status = "motion seen, same side"
        else:
            score_status = "motion seen, waiting for crossing"

        return {
            "count": self.count,
            "event_detected": event_detected,
            "motion_scored": event_detected,
            "score_status": score_status,
            "crossing_method": self.crossing_method,
            "leading_edge_margin": self.leading_edge_margin,
            "current_side": current_side,
            "previous_side": self.previous_side,
            "armed_for_crossing": self.armed_for_crossing,
            "blob_center": blob_center,
            "blob_area": blob_area,
            "left_edge": left_edge,
            "right_edge": right_edge,
            "bbox": bbox,
            "zone_bounds": zone_bounds,
            "cooldown_remaining": cooldown_remaining,
        }

    def draw_debug_overlay(self, frame, result):
        annotated = frame.copy()
        frame_height, frame_width = annotated.shape[:2]

        x1 = self.partition_x - (self.crossing_zone_width // 2)
        x2 = self.partition_x + (self.crossing_zone_width // 2)
        x1 = max(0, x1)
        x2 = min(frame_width, x2)

        left_dead_zone = int(round(self.partition_x - (self.dead_zone_width / 2.0)))
        right_dead_zone = int(round(self.partition_x + (self.dead_zone_width / 2.0)))
        left_margin_line = int(round(self.partition_x - self.leading_edge_margin))
        right_margin_line = int(round(self.partition_x + self.leading_edge_margin))

        cv2.line(annotated, (self.partition_x, 0), (self.partition_x, frame_height - 1), (0, 0, 255), 2)
        cv2.rectangle(annotated, (x1, 0), (x2, frame_height - 1), (255, 0, 0), 2)
        cv2.line(
            annotated,
            (max(left_dead_zone, 0), 0),
            (max(left_dead_zone, 0), frame_height - 1),
            (0, 255, 255),
            2,
        )
        cv2.line(
            annotated,
            (min(right_dead_zone, frame_width - 1), 0),
            (min(right_dead_zone, frame_width - 1), frame_height - 1),
            (0, 255, 255),
            2,
        )
        cv2.line(
            annotated,
            (max(left_margin_line, 0), 0),
            (max(left_margin_line, 0), frame_height - 1),
            (255, 180, 0),
            2,
        )
        cv2.line(
            annotated,
            (min(right_margin_line, frame_width - 1), 0),
            (min(right_margin_line, frame_width - 1), frame_height - 1),
            (255, 180, 0),
            2,
        )

        if result["bbox"] is not None:
            x, y, w, h = result["bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (50, 220, 50), 2)

        if result["blob_center"] is not None:
            cx, cy = result["blob_center"]
            cv2.circle(annotated, (cx, cy), 5, (50, 220, 50), -1)
        if result["left_edge"] is not None:
            cv2.line(
                annotated,
                (int(result["left_edge"]), 0),
                (int(result["left_edge"]), frame_height - 1),
                (120, 255, 120),
                1,
            )
        if result["right_edge"] is not None:
            cv2.line(
                annotated,
                (int(result["right_edge"]), 0),
                (int(result["right_edge"]), frame_height - 1),
                (120, 255, 120),
                1,
            )

        score_status = result.get("score_status", "unknown")
        scored_this_frame = bool(result.get("motion_scored", False))
        status_color = (0, 220, 0) if scored_this_frame else (0, 200, 255)

        info_lines = [
            f"Count: {result['count']}",
            f"Scored this frame: {'YES' if scored_this_frame else 'NO'}",
            f"Status: {score_status}",
            f"Crossing method: {result['crossing_method']}",
            f"Armed: {'YES' if result['armed_for_crossing'] else 'NO'}",
            f"Current side: {result['current_side']}",
            f"Left edge: {result['left_edge']}",
            f"Right edge: {result['right_edge']}",
            f"Blob area: {result['blob_area']:.0f}",
            f"Cooldown: {result['cooldown_remaining']}",
        ]
        y = 30
        for line in info_lines:
            cv2.putText(annotated, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(annotated, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            y += 30

        if result["event_detected"]:
            cv2.putText(
                annotated,
                "CROSSING!",
                (20, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

        badge_text = f"SCORE: {'YES' if scored_this_frame else 'NO'}"
        text_size, _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)
        badge_x = max(frame_width - text_size[0] - 40, 20)
        badge_y = 45
        cv2.rectangle(
            annotated,
            (badge_x - 12, badge_y - 28),
            (badge_x + text_size[0] + 12, badge_y + 12),
            status_color,
            -1,
        )
        cv2.putText(
            annotated,
            badge_text,
            (badge_x, badge_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        return annotated


def resize_frame(frame, resize_width=None, resize_height=None):
    if resize_width is None and resize_height is None:
        return frame

    height, width = frame.shape[:2]
    if resize_width is None:
        scale = float(resize_height) / float(height)
        resize_width = int(round(width * scale))
    elif resize_height is None:
        scale = float(resize_width) / float(width)
        resize_height = int(round(height * scale))

    return cv2.resize(frame, (int(resize_width), int(resize_height)), interpolation=cv2.INTER_AREA)


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
        "--crossing-method",
        choices=["center", "leading_edge"],
        default="leading_edge",
        help="How to decide that a crossing happened.",
    )
    parser.add_argument(
        "--leading-edge-margin",
        type=int,
        default=10,
        help="How far the leading edge must pass the partition before it counts.",
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
        crossing_method=args.crossing_method,
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
            resized_frame = resize_frame(
                frame,
                resize_width=args.resize_width,
                resize_height=args.resize_height,
            )
            raw_mask = background_subtractor.apply(resized_frame)
            cleaned_mask = clean_motion_mask(raw_mask)
            result = counter.update(cleaned_mask, frame_idx=frame_index)
            annotated_frame = counter.draw_debug_overlay(resized_frame, result)

            if args.save_output and writer is None:
                output_path = Path(args.save_output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
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
    print(f"Crossing method: {counter.crossing_method}")
    print(f"Leading edge margin: {counter.leading_edge_margin}")
    print(f"Armed for crossing: {counter.armed_for_crossing}")
    print(f"Last left edge: {counter.current_left_edge}")
    print(f"Last right edge: {counter.current_right_edge}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
