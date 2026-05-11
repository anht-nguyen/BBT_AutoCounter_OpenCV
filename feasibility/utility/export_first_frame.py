from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2


def choose_video_file() -> Path | None:
    """Open a file picker so the user can choose a video."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    root = tk.Tk()
    root.withdraw()
    root.update()

    file_path = filedialog.askopenfilename(
        title="Choose a video",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()
    return Path(file_path) if file_path else None


def export_first_frame(video_path: Path, output_path: Path | None = None) -> Path:
    """Read the first frame from a video and save it as an image."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        output_path = video_path.with_name(f"{video_path.stem}_first_frame.jpg")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    success, frame = capture.read()
    capture.release()

    if not success or frame is None:
        raise RuntimeError(f"Could not read first frame: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Could not write image: {output_path}")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the first frame from a selected video."
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="Path to the video file. If omitted, a file picker opens.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output image path. Defaults to <video>_first_frame.jpg.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    video_path = Path(args.video) if args.video else choose_video_file()
    if video_path is None:
        print("No video selected.", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else None

    try:
        saved_path = export_first_frame(video_path, output_path)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Saved first frame to: {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
