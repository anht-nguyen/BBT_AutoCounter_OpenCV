from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bbt_autocounter.pipeline import BBTScoringPipeline, PipelineConfig
from bbt_autocounter.ui import open_writer, resize_for_display, resize_frame


FEASIBILITY_ROOT = PROJECT_ROOT / "feasibility"
VIDEO_PATH = FEASIBILITY_ROOT / "data" / "videos" / "raw" / "BBT-ground_truth.mp4"
OUTPUT_DIR = FEASIBILITY_ROOT / "data" / "videos" / "annotated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score BBT transfers using motion, hand confirmation, and object confirmation.")
    parser.add_argument("--video", type=Path, default=VIDEO_PATH)
    parser.add_argument("--model-asset-path", type=Path, default=None)
    parser.add_argument("--save-output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    pipeline = BBTScoringPipeline(PipelineConfig(partition_x=640, model_asset_path=None if args.model_asset_path is None else str(args.model_asset_path)))
    output_path = args.save_output or (OUTPUT_DIR / f"{args.video.stem}_scoring_with_hand_confirmation.mp4")
    writer = None
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break
            frame_index += 1
            resized = resize_frame(frame, resize_width=1280, resize_height=720)
            artifacts = pipeline.process_frame(resized, frame_index)
            if writer is None:
                writer = open_writer(output_path, (artifacts.comparison_view.shape[1], artifacts.comparison_view.shape[0]), fps)
            writer.write(artifacts.comparison_view)
            cv2.imshow("BBT Scoring With Hand Confirmation", resize_for_display(artifacts.comparison_view, 0.60))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                cv2.waitKey(0)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        pipeline.close()
        cv2.destroyAllWindows()
    print(f"Processed video: {args.video}")
    print(f"Frames processed: {frame_index}")
    print(f"Saved annotated video: {output_path}")
    print(f"Combined score count: {pipeline.summary.combined_count}")
    print(f"Total motion events: {pipeline.summary.total_motion_events}")
    print(f"Hand-confirmed events: {pipeline.summary.total_hand_confirmed_events}")
    print(f"Rejected motion events without hand confirmation: {pipeline.summary.rejected_motion_events_without_hand}")
    print(f"Rejected motion events without object confirmation: {pipeline.summary.rejected_motion_events_without_object}")
    print(f"Motion detector candidate crossings: {pipeline.counter.total_candidate_crossings}")
    print(f"Motion detector confirmed crossings: {pipeline.counter.confirmed_crossings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
