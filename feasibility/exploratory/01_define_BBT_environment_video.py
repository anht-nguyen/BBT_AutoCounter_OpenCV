from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEASIBILITY_ROOT = PROJECT_ROOT / "feasibility"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ANNOTATION_DIR = FEASIBILITY_ROOT / "data" / "images" / "annotations"
MASK_DIR = ANNOTATION_DIR / "masks"
ENVIRONMENT_JSON = ANNOTATION_DIR / "BBT_environment.json"
VIDEO_PATH = FEASIBILITY_ROOT / "data" / "videos" / "raw" / "BBT-ground_truth.mp4"
OUTPUT_DIR = FEASIBILITY_ROOT / "data" / "videos" / "annotated"

WINDOW_NAME = "BBT Environment Applied To Video"
DISPLAY_SCALE = 0.7
PREVIEW_MAX_FRAMES = None
WRITE_OUTPUT_VIDEO = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview the saved BBT environment overlays on a raw video."
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
    return OUTPUT_DIR / f"{video_path.stem}_environment_overlay.mp4"


@dataclass
class VideoEnvironment:
    partition_line: list[tuple[int, int]]
    start_polygon: list[tuple[int, int]]
    target_polygon: list[tuple[int, int]]
    crossing_zone_polygon: list[tuple[int, int]]
    start_side: str
    crossing_zone_width: int


def load_environment(environment_path: Path) -> VideoEnvironment:
    payload = json.loads(environment_path.read_text(encoding="utf-8"))
    return VideoEnvironment(
        partition_line=[tuple(point) for point in payload["partition_line"]],
        start_polygon=[tuple(point) for point in payload["start_polygon"]],
        target_polygon=[tuple(point) for point in payload["target_polygon"]],
        crossing_zone_polygon=[tuple(point) for point in payload["crossing_zone_polygon"]],
        start_side=payload["start_side"],
        crossing_zone_width=int(payload["crossing_zone_width"]),
    )


def load_mask(mask_path: Path, frame_size: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {mask_path}")

    frame_width, frame_height = frame_size
    if (mask.shape[1], mask.shape[0]) != (frame_width, frame_height):
        mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

    return (mask > 0).astype(np.uint8) * 255


def blend_mask(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    overlay = frame.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_polygon_outline(
    frame: np.ndarray,
    polygon: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.polylines(frame, [np.asarray(polygon, dtype=np.int32)], True, color, thickness)


def draw_partition_line(
    frame: np.ndarray,
    partition_line: list[tuple[int, int]],
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 3,
) -> None:
    cv2.line(frame, partition_line[0], partition_line[1], color, thickness)


def draw_label(
    frame: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = anchor
    origin = (max(x + 10, 15), max(y + 30, 30))
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)


def annotate_frame(
    frame: np.ndarray,
    environment: VideoEnvironment,
    masks: dict[str, np.ndarray],
    frame_index: int,
) -> np.ndarray:
    annotated = frame.copy()
    annotated = blend_mask(annotated, masks["start_side"], (70, 180, 70), 0.18)
    annotated = blend_mask(annotated, masks["target_side"], (40, 170, 240), 0.18)
    annotated = blend_mask(annotated, masks["crossing_zone"], (255, 140, 30), 0.22)

    draw_polygon_outline(annotated, environment.start_polygon, (70, 180, 70))
    draw_polygon_outline(annotated, environment.target_polygon, (40, 170, 240))
    draw_polygon_outline(annotated, environment.crossing_zone_polygon, (255, 140, 30))
    draw_partition_line(annotated, environment.partition_line)

    draw_label(annotated, "Start side", environment.start_polygon[0], (70, 180, 70))
    draw_label(annotated, "Target side", environment.target_polygon[1], (40, 170, 240))
    draw_label(annotated, "Crossing zone", environment.crossing_zone_polygon[0], (255, 140, 30))

    info_lines = [
        f"Frame: {frame_index}",
        f"Start side: {environment.start_side}",
        f"Crossing zone width: {environment.crossing_zone_width}px",
    ]
    y = 35
    for line in info_lines:
        cv2.putText(annotated, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(annotated, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y += 32

    return annotated


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


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    environment = load_environment(ENVIRONMENT_JSON)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_size = (frame_width, frame_height)

    masks = {
        "start_side": load_mask(MASK_DIR / "start_side_mask.png", frame_size),
        "target_side": load_mask(MASK_DIR / "target_side_mask.png", frame_size),
        "crossing_zone": load_mask(MASK_DIR / "crossing_zone_mask.png", frame_size),
    }

    output_video = args.output_video.expanduser().resolve() if args.output_video else build_output_video_path(video_path)
    writer = open_writer(output_video, frame_size, fps) if WRITE_OUTPUT_VIDEO else None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break

            frame_index += 1
            annotated = annotate_frame(frame, environment, masks, frame_index)

            if writer is not None:
                writer.write(annotated)

            display_frame = annotated
            if DISPLAY_SCALE != 1.0:
                display_frame = cv2.resize(
                    annotated,
                    None,
                    fx=DISPLAY_SCALE,
                    fy=DISPLAY_SCALE,
                    interpolation=cv2.INTER_AREA,
                )

            cv2.imshow(WINDOW_NAME, display_frame)
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

    print(f"Applied masks from: {MASK_DIR}")
    print(f"Processed video: {video_path}")
    if writer is not None:
        print(f"Saved annotated video to: {output_video}")
    print(f"Frames processed: {frame_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
