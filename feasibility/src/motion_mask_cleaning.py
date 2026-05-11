from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def ensure_odd_kernel_size(value, minimum=1) -> int:
    """Convert a kernel size to a safe odd integer."""
    corrected_value = int(value)
    corrected_value = max(corrected_value, int(minimum))
    if corrected_value % 2 == 0:
        corrected_value += 1
    return corrected_value


def clean_motion_mask(
    raw_mask,
    blur_kernel_size=5,
    threshold_value=200,
    morph_kernel_size=5,
    opening_iterations=1,
    closing_iterations=1,
):
    """Clean a raw foreground mask so moving regions become easier to inspect."""
    blur_kernel_size = ensure_odd_kernel_size(blur_kernel_size, minimum=1)
    morph_kernel_size = ensure_odd_kernel_size(morph_kernel_size, minimum=1)

    # Blur helps reduce tiny noisy pixels before thresholding.
    blurred_mask = cv2.GaussianBlur(raw_mask, (blur_kernel_size, blur_kernel_size), 0)

    # Thresholding converts the mask into a clear black/white image.
    _, binary_mask = cv2.threshold(
        blurred_mask,
        int(threshold_value),
        255,
        cv2.THRESH_BINARY,
    )

    # A small square kernel is enough for simple opening and closing.
    kernel = np.ones((morph_kernel_size, morph_kernel_size), dtype=np.uint8)

    # Opening removes small isolated white specks.
    opened_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=max(int(opening_iterations), 0),
    )

    # Closing fills small holes inside a moving blob.
    cleaned_mask = cv2.morphologyEx(
        opened_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=max(int(closing_iterations), 0),
    )
    return cleaned_mask


def filter_contours_by_area(binary_mask, min_area=500):
    """Keep only contours that are large enough to matter."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_info_list = []

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

        contour_info_list.append(
            {
                "contour": contour,
                "area": area,
                "bbox": (x, y, w, h),
                "center": (cx, cy),
            }
        )

    contour_info_list.sort(key=lambda item: item["area"], reverse=True)
    return contour_info_list


def draw_contour_debug(frame, contour_info_list):
    """Draw simple contour overlays for debugging."""
    debug_frame = frame.copy()

    for contour_info in contour_info_list:
        x, y, w, h = contour_info["bbox"]
        cx, cy = contour_info["center"]
        area = contour_info["area"]

        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (50, 220, 50), 2)
        cv2.circle(debug_frame, (cx, cy), 5, (50, 220, 50), -1)
        cv2.putText(
            debug_frame,
            f"area: {area:.0f}",
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (50, 220, 50),
            2,
            cv2.LINE_AA,
        )

    return debug_frame


def resize_frame(frame, resize_width=None, resize_height=None):
    """Resize the frame if either target dimension is provided."""
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


def mask_to_bgr(mask):
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean a motion mask and inspect valid motion contours in a video."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--blur-kernel-size", type=int, default=5, help="Gaussian blur kernel size.")
    parser.add_argument("--threshold-value", type=int, default=200, help="Binary threshold value.")
    parser.add_argument("--morph-kernel-size", type=int, default=5, help="Morphology kernel size.")
    parser.add_argument("--opening-iterations", type=int, default=1, help="Opening iterations.")
    parser.add_argument("--closing-iterations", type=int, default=1, help="Closing iterations.")
    parser.add_argument("--min-area", type=float, default=500, help="Minimum contour area to keep.")
    parser.add_argument("--resize-width", type=int, default=None, help="Optional output frame width.")
    parser.add_argument("--resize-height", type=int, default=None, help="Optional output frame height.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Could not find video: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=False,
    )

    cv2.namedWindow("Original Frame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Raw Motion Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Cleaned Motion Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Contour Debug Frame", cv2.WINDOW_NORMAL)

    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break

            resized_frame = resize_frame(
                frame,
                resize_width=args.resize_width,
                resize_height=args.resize_height,
            )
            raw_mask = background_subtractor.apply(resized_frame)
            cleaned_mask = clean_motion_mask(
                raw_mask,
                blur_kernel_size=args.blur_kernel_size,
                threshold_value=args.threshold_value,
                morph_kernel_size=args.morph_kernel_size,
                opening_iterations=args.opening_iterations,
                closing_iterations=args.closing_iterations,
            )
            contour_info_list = filter_contours_by_area(cleaned_mask, min_area=args.min_area)
            contour_debug_frame = draw_contour_debug(resized_frame, contour_info_list)

            cv2.imshow("Original Frame", resized_frame)
            cv2.imshow("Raw Motion Mask", raw_mask)
            cv2.imshow("Cleaned Motion Mask", cleaned_mask)
            cv2.imshow("Contour Debug Frame", contour_debug_frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
