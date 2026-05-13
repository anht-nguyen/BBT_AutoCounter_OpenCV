from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .environment import BBTEnvironment, polygon_to_mask
from .ui import add_panel_label, mask_to_bgr, overlay_mask


@dataclass(frozen=True)
class MotionMaskCleanerConfig:
    blur_kernel_size: int = 5
    threshold_value: int = 200
    morph_kernel_size: int = 5
    opening_iterations: int = 1
    closing_iterations: int = 1


@dataclass(frozen=True)
class MotionDetectorConfig:
    history: int = 500
    var_threshold: float = 50.0
    detect_shadows: bool = False
    min_area: int = 500
    learning_rate: float = -1.0
    cleaner: MotionMaskCleanerConfig = field(default_factory=MotionMaskCleanerConfig)


@dataclass
class ContourInfo:
    contour: np.ndarray
    area: float
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]


@dataclass
class MotionBlob:
    contour: np.ndarray
    area: float
    bounding_box: tuple[int, int, int, int]
    centroid: tuple[int, int]


@dataclass
class MotionDetectionResult:
    frame_index: int
    foreground_mask: np.ndarray
    cleaned_mask: np.ndarray
    crossing_zone_mask: np.ndarray
    crossing_zone_foreground: np.ndarray
    blob: MotionBlob | None
    contours: list[np.ndarray]

    def __getitem__(self, key: str):
        return getattr(self, key)


def ensure_odd_kernel_size(value: int, minimum: int = 1) -> int:
    corrected_value = max(int(value), int(minimum))
    if corrected_value % 2 == 0:
        corrected_value += 1
    return corrected_value


def clean_motion_mask(raw_mask: np.ndarray, config: MotionMaskCleanerConfig | None = None, **overrides: int) -> np.ndarray:
    settings = config or MotionMaskCleanerConfig()
    blur_kernel_size = ensure_odd_kernel_size(overrides.get("blur_kernel_size", settings.blur_kernel_size))
    morph_kernel_size = ensure_odd_kernel_size(overrides.get("morph_kernel_size", settings.morph_kernel_size))
    threshold_value = int(overrides.get("threshold_value", settings.threshold_value))
    opening_iterations = max(int(overrides.get("opening_iterations", settings.opening_iterations)), 0)
    closing_iterations = max(int(overrides.get("closing_iterations", settings.closing_iterations)), 0)

    blurred_mask = cv2.GaussianBlur(raw_mask, (blur_kernel_size, blur_kernel_size), 0)
    _, binary_mask = cv2.threshold(blurred_mask, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = np.ones((morph_kernel_size, morph_kernel_size), dtype=np.uint8)
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=opening_iterations)
    return cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)


def filter_contours_by_area(binary_mask: np.ndarray, min_area: float = 500) -> list[ContourInfo]:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_info_list: list[ContourInfo] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(min_area):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if abs(float(moments["m00"])) > 1e-8:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx = x + (w // 2)
            cy = y + (h // 2)
        contour_info_list.append(ContourInfo(contour=contour, area=area, bbox=(x, y, w, h), center=(cx, cy)))
    contour_info_list.sort(key=lambda item: item.area, reverse=True)
    return contour_info_list


def draw_contour_debug(frame: np.ndarray, contour_info_list: list[ContourInfo]) -> np.ndarray:
    debug_frame = frame.copy()
    for contour_info in contour_info_list:
        x, y, w, h = contour_info.bbox
        cx, cy = contour_info.center
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (50, 220, 50), 2)
        cv2.circle(debug_frame, (cx, cy), 5, (50, 220, 50), -1)
        cv2.putText(
            debug_frame,
            f"area: {contour_info.area:.0f}",
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (50, 220, 50),
            2,
            cv2.LINE_AA,
        )
    return debug_frame


class CrossingZoneMotionDetector:
    def __init__(self, environment: BBTEnvironment, config: MotionDetectorConfig | None = None) -> None:
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
        cleaned_mask = clean_motion_mask(foreground_mask, config=self.config.cleaner)
        crossing_zone_mask = self._get_crossing_zone_mask(frame.shape)
        crossing_zone_foreground = cv2.bitwise_and(cleaned_mask, crossing_zone_mask)
        blob, contours = self._find_largest_blob(crossing_zone_foreground)
        return MotionDetectionResult(
            frame_index=frame_index,
            foreground_mask=foreground_mask,
            cleaned_mask=cleaned_mask,
            crossing_zone_mask=crossing_zone_mask,
            crossing_zone_foreground=crossing_zone_foreground,
            blob=blob,
            contours=contours,
        )

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
            centroid = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
        else:
            centroid = (x + (w // 2), y + (h // 2))
        return MotionBlob(largest, float(cv2.contourArea(largest)), (x, y, w, h), centroid), valid_contours


def annotate_motion_frame(
    frame: np.ndarray,
    environment: BBTEnvironment,
    result: MotionDetectionResult,
    show_all_contours: bool = False,
) -> np.ndarray:
    annotated = overlay_mask(frame.copy(), result.crossing_zone_mask, (255, 140, 30), alpha=0.18)
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
    return annotated


def build_motion_debug_panel(frame: np.ndarray, result: MotionDetectionResult) -> np.ndarray:
    zone_overlay = mask_to_bgr(result.crossing_zone_foreground)
    if result.blob is not None:
        cv2.drawContours(zone_overlay, [result.blob.contour], -1, (0, 255, 0), 2)
        x, y, w, h = result.blob.bounding_box
        cv2.rectangle(zone_overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return np.hstack(
        [
            add_panel_label(frame, "Original"),
            add_panel_label(mask_to_bgr(result.cleaned_mask), "Motion Mask"),
            add_panel_label(zone_overlay, "Crossing Zone Motion"),
        ]
    )
