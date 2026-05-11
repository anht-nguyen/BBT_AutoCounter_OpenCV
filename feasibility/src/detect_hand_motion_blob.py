from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


Point = tuple[int, int]


@dataclass
class MotionEnvironment:
    partition_line: list[Point]
    start_polygon: list[Point]
    target_polygon: list[Point]
    crossing_zone_polygon: list[Point]
    start_side: str
    crossing_zone_width: int


@dataclass
class MotionBlob:
    contour: np.ndarray
    area: float
    bounding_box: tuple[int, int, int, int]
    centroid: Point


@dataclass
class MotionDetectionResult:
    frame_index: int
    foreground_mask: np.ndarray
    crossing_zone_mask: np.ndarray
    crossing_zone_foreground: np.ndarray
    filtered_mask: np.ndarray
    blob: MotionBlob | None
    contours: list[np.ndarray] = field(default_factory=list)


@dataclass
class MotionDetectorConfig:
    history: int = 500
    var_threshold: float = 50.0
    detect_shadows: bool = False
    blur_kernel_size: tuple[int, int] = (5, 5)
    threshold_value: int = 200
    morphology_kernel_size: tuple[int, int] = (5, 5)
    min_area: int = 500
    learning_rate: float = -1.0

    def morphology_kernel(self) -> np.ndarray:
        return np.ones(self.morphology_kernel_size, dtype=np.uint8)


def load_environment(environment_path: Path) -> MotionEnvironment:
    payload = json.loads(environment_path.read_text(encoding="utf-8"))
    return MotionEnvironment(
        partition_line=[tuple(point) for point in payload["partition_line"]],
        start_polygon=[tuple(point) for point in payload["start_polygon"]],
        target_polygon=[tuple(point) for point in payload["target_polygon"]],
        crossing_zone_polygon=[tuple(point) for point in payload["crossing_zone_polygon"]],
        start_side=str(payload["start_side"]),
        crossing_zone_width=int(payload["crossing_zone_width"]),
    )


def polygon_to_mask(frame_shape: tuple[int, int] | tuple[int, int, int], polygon: list[Point]) -> np.ndarray:
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
    return mask


class HandMotionBlobDetector:
    def __init__(self, environment: MotionEnvironment, config: MotionDetectorConfig | None = None) -> None:
        self.environment = environment
        self.config = config or MotionDetectorConfig()
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.var_threshold,
            detectShadows=self.config.detect_shadows,
        )
        self._crossing_zone_mask: np.ndarray | None = None

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> MotionDetectionResult:
        foreground_mask = self._subtractor.apply(frame, learningRate=self.config.learning_rate)
        filtered_mask = self._clean_foreground_mask(foreground_mask)
        crossing_zone_mask = self._get_crossing_zone_mask(frame.shape)
        crossing_zone_foreground = cv2.bitwise_and(filtered_mask, crossing_zone_mask)
        blob, contours = self._find_largest_blob(crossing_zone_foreground)
        return MotionDetectionResult(
            frame_index=frame_index,
            foreground_mask=foreground_mask,
            crossing_zone_mask=crossing_zone_mask,
            crossing_zone_foreground=crossing_zone_foreground,
            filtered_mask=filtered_mask,
            blob=blob,
            contours=contours,
        )

    def _clean_foreground_mask(self, foreground_mask: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(foreground_mask, self.config.blur_kernel_size, 0)
        _, binary = cv2.threshold(blurred, self.config.threshold_value, 255, cv2.THRESH_BINARY)
        kernel = self.config.morphology_kernel()
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed

    def _get_crossing_zone_mask(self, frame_shape: tuple[int, int, int]) -> np.ndarray:
        if self._crossing_zone_mask is None or self._crossing_zone_mask.shape != frame_shape[:2]:
            self._crossing_zone_mask = polygon_to_mask(frame_shape, self.environment.crossing_zone_polygon)
        return self._crossing_zone_mask

    def _find_largest_blob(self, mask: np.ndarray) -> tuple[MotionBlob | None, list[np.ndarray]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [contour for contour in contours if cv2.contourArea(contour) >= self.config.min_area]
        if not valid_contours:
            return None, []

        largest = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        moments = cv2.moments(largest)
        if abs(float(moments["m00"])) > 1e-8:
            centroid = (
                int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"]),
            )
        else:
            centroid = (x + (w // 2), y + (h // 2))

        blob = MotionBlob(
            contour=largest,
            area=float(cv2.contourArea(largest)),
            bounding_box=(x, y, w, h),
            centroid=centroid,
        )
        return blob, valid_contours


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def overlay_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.25,
) -> np.ndarray:
    overlay = frame.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def annotate_motion_frame(
    frame: np.ndarray,
    environment: MotionEnvironment,
    result: MotionDetectionResult,
    show_all_contours: bool = False,
) -> np.ndarray:
    annotated = frame.copy()
    annotated = overlay_mask(annotated, result.crossing_zone_mask, (255, 140, 30), alpha=0.18)

    crossing_polygon = np.asarray(environment.crossing_zone_polygon, dtype=np.int32)
    partition_line = np.asarray(environment.partition_line, dtype=np.int32)
    cv2.polylines(annotated, [crossing_polygon], True, (255, 140, 30), 2)
    cv2.line(annotated, tuple(partition_line[0]), tuple(partition_line[1]), (0, 0, 255), 2)

    if show_all_contours:
        cv2.drawContours(annotated, result.contours, -1, (80, 220, 220), 2)

    if result.blob is not None:
        x, y, w, h = result.blob.bounding_box
        cv2.drawContours(annotated, [result.blob.contour], -1, (50, 220, 50), 2)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (50, 220, 50), 2)
        cv2.circle(annotated, result.blob.centroid, 5, (50, 220, 50), -1)
        cv2.putText(
            annotated,
            f"motion area: {result.blob.area:.0f}",
            (max(x, 20), max(y - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 220, 50),
            2,
            cv2.LINE_AA,
        )

    status = "motion detected" if result.blob is not None else "no valid motion blob"
    lines = [
        f"Frame: {result.frame_index}",
        f"Status: {status}",
        f"Crossing zone width: {environment.crossing_zone_width}px",
    ]
    if result.blob is not None:
        x, y, w, h = result.blob.bounding_box
        lines.append(f"Blob bbox: x={x}, y={y}, w={w}, h={h}")

    y = 32
    for line in lines:
        cv2.putText(annotated, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(annotated, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28

    return annotated


def build_debug_panel(frame: np.ndarray, result: MotionDetectionResult) -> np.ndarray:
    annotated_mask = mask_to_bgr(result.filtered_mask)
    zone_overlay = mask_to_bgr(result.crossing_zone_foreground)

    if result.blob is not None:
        cv2.drawContours(zone_overlay, [result.blob.contour], -1, (0, 255, 0), 2)
        x, y, w, h = result.blob.bounding_box
        cv2.rectangle(zone_overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    panels = [
        _label_panel(frame, "Original"),
        _label_panel(annotated_mask, "Motion Mask"),
        _label_panel(zone_overlay, "Crossing Zone Mask"),
    ]
    return np.hstack(panels)


def _label_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = image.copy()
    cv2.putText(panel, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(panel, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return panel
