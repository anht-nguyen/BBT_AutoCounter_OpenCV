from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def resize_frame(
    frame: np.ndarray,
    resize_width: int | None = None,
    resize_height: int | None = None,
) -> np.ndarray:
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


def resize_for_display(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def add_panel_label(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    draw_text_block(labeled, [label], origin=(16, 28), font_scale=0.8, line_spacing=28)
    return labeled


def draw_text_block(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (20, 32),
    font_scale: float = 0.7,
    line_spacing: int = 28,
    color: tuple[int, int, int] = (255, 255, 255),
    outline_color: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
    outline_thickness: int = 4,
) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            outline_color,
            outline_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_spacing


def stack_views(rows: list[list[np.ndarray]]) -> np.ndarray:
    return np.vstack([np.hstack(row) for row in rows])


def overlay_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.25,
) -> np.ndarray:
    overlay = frame.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


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
