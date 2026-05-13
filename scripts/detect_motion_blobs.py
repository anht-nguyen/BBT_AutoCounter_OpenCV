from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bbt_autocounter.environment import load_environment
from bbt_autocounter.motion import CrossingZoneMotionDetector, MotionDetectorConfig, annotate_motion_frame, build_motion_debug_panel
from bbt_autocounter.ui import open_writer, resize_for_display


FEASIBILITY_ROOT = PROJECT_ROOT / "feasibility"
ENVIRONMENT_JSON = FEASIBILITY_ROOT / "data" / "images" / "annotations" / "BBT_environment.json"
VIDEO_PATH = FEASIBILITY_ROOT / "data" / "videos" / "raw" / "BBT-ground_truth.mp4"
OUTPUT_DIR = FEASIBILITY_ROOT / "data" / "videos" / "annotated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run crossing-zone motion detection on a raw BBT video.")
    parser.add_argument("--video", type=Path, default=VIDEO_PATH)
    parser.add_argument("--output-video", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    environment = load_environment(ENVIRONMENT_JSON)
    detector = CrossingZoneMotionDetector(environment=environment, config=MotionDetectorConfig())
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    output_path = args.output_video or (OUTPUT_DIR / f"{args.video.stem}_motion_detection.mp4")
    writer = None
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break
            frame_index += 1
            result = detector.detect(frame, frame_index=frame_index)
            annotated = annotate_motion_frame(frame, environment, result, show_all_contours=True)
            debug_panel = build_motion_debug_panel(frame, result)
            if writer is None:
                writer = open_writer(output_path, (annotated.shape[1], annotated.shape[0]), fps)
            writer.write(annotated)
            cv2.imshow("BBT Motion Detection", resize_for_display(annotated, 0.55))
            cv2.imshow("BBT Motion Debug", resize_for_display(debug_panel, 0.55))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
