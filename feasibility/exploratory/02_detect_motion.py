from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEASIBILITY_ROOT = PROJECT_ROOT / "feasibility"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feasibility.src.detect_hand_motion_blob import HandMotionBlobDetector
from feasibility.src.detect_hand_motion_blob import MotionDetectorConfig
from feasibility.src.detect_hand_motion_blob import annotate_motion_frame
from feasibility.src.detect_hand_motion_blob import build_debug_panel
from feasibility.src.detect_hand_motion_blob import load_environment


ENVIRONMENT_JSON = FEASIBILITY_ROOT / "data" / "images" / "annotations" / "BBT_environment.json"
VIDEO_PATH = FEASIBILITY_ROOT / "data" / "videos" / "raw" / "BBT-ground_truth.mp4"
OUTPUT_DIR = FEASIBILITY_ROOT / "data" / "videos" / "annotated"

WINDOW_NAME = "BBT Motion Detection"
DEBUG_WINDOW_NAME = "BBT Motion Debug"
DISPLAY_SCALE = 0.55
PREVIEW_MAX_FRAMES = None
WRITE_OUTPUT_VIDEO = True
SHOW_ALL_CONTOURS = True

CONFIG = MotionDetectorConfig(
    history=500,
    var_threshold=50,
    detect_shadows=False,
    blur_kernel_size=(5, 5),
    threshold_value=200,
    morphology_kernel_size=(5, 5),
    min_area=500,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exploratory motion detection on a raw BBT video."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=VIDEO_PATH,
        help="Path to the raw input video.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Optional path for the annotated output video.",
    )
    return parser.parse_args()


def build_output_video_path(video_path: Path) -> Path:
    return OUTPUT_DIR / f"{video_path.stem}_motion_detection.mp4"


def open_writer(output_path: Path, frame_size: tuple[int, int], fps: float) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {output_path}")
    return writer


def resize_for_display(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    environment = load_environment(ENVIRONMENT_JSON)
    detector = HandMotionBlobDetector(environment=environment, config=CONFIG)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_size = (frame_width, frame_height)

    output_video = args.output_video.expanduser().resolve() if args.output_video else build_output_video_path(video_path)
    writer = open_writer(output_video, frame_size, fps) if WRITE_OUTPUT_VIDEO else None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break

            frame_index += 1
            result = detector.detect(frame, frame_index=frame_index)
            annotated = annotate_motion_frame(
                frame=frame,
                environment=environment,
                result=result,
                show_all_contours=SHOW_ALL_CONTOURS,
            )
            debug_panel = build_debug_panel(frame, result)

            if writer is not None:
                writer.write(annotated)

            cv2.imshow(WINDOW_NAME, resize_for_display(annotated, DISPLAY_SCALE))
            cv2.imshow(DEBUG_WINDOW_NAME, resize_for_display(debug_panel, DISPLAY_SCALE))

            key = cv2.waitKey(max(int(1000 / max(fps, 1.0)), 1)) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                cv2.waitKey(0)

            if PREVIEW_MAX_FRAMES is not None and frame_index >= PREVIEW_MAX_FRAMES:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"Environment: {ENVIRONMENT_JSON}")
    print(f"Video: {video_path}")
    print(f"Frames processed: {frame_index}")
    if writer is not None:
        print(f"Saved motion detection preview to: {output_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
