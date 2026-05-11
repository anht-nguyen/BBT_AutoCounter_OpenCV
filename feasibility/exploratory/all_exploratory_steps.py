from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEASIBILITY_ROOT = PROJECT_ROOT / "feasibility"
RAW_VIDEO_DIR = FEASIBILITY_ROOT / "data" / "videos" / "raw"


@dataclass(frozen=True)
class ExploratoryStep:
    step_id: str
    script_path: Path
    description: str


STEPS: tuple[ExploratoryStep, ...] = (
    ExploratoryStep(
        step_id="01",
        script_path=Path(__file__).with_name("01_define_BBT_environment_video.py"),
        description="Apply environment overlays",
    ),
    ExploratoryStep(
        step_id="02",
        script_path=Path(__file__).with_name("02_detect_motion.py"),
        description="Detect motion blobs",
    ),
    ExploratoryStep(
        step_id="03",
        script_path=Path(__file__).with_name("03_clean_motion_mask.py"),
        description="Inspect cleaned motion masks",
    ),
    ExploratoryStep(
        step_id="04",
        script_path=Path(__file__).with_name("04_detect_crossing_event.py"),
        description="Detect crossing events",
    ),
)


def list_raw_videos(raw_video_dir: Path) -> list[Path]:
    if not raw_video_dir.exists():
        raise FileNotFoundError(f"Raw video directory not found: {raw_video_dir}")

    videos = sorted(
        path for path in raw_video_dir.iterdir() if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )
    if not videos:
        raise FileNotFoundError(f"No raw video files found in: {raw_video_dir}")
    return videos


def prompt_for_videos(videos: list[Path]) -> list[Path]:
    print("Available raw videos:")
    for index, video_path in enumerate(videos, start=1):
        print(f"  {index}. {video_path.name}")

    print()
    print("Select videos by number separated with commas, or type 'all'.")

    while True:
        selection = input("Selection: ").strip()
        if not selection:
            print("Enter at least one selection.")
            continue

        if selection.lower() == "all":
            return videos

        try:
            indexes = []
            for part in selection.split(","):
                value = int(part.strip())
                if value < 1 or value > len(videos):
                    raise ValueError
                indexes.append(value - 1)
        except ValueError:
            print("Invalid selection. Use values from the list, for example: 1 or 1,3")
            continue

        unique_indexes = []
        for index in indexes:
            if index not in unique_indexes:
                unique_indexes.append(index)
        return [videos[index] for index in unique_indexes]


def run_step(step: ExploratoryStep, video_path: Path) -> None:
    command = [sys.executable, str(step.script_path), "--video", str(video_path)]
    print()
    print(f"[{step.step_id}] {step.description}")
    print(f"Video: {video_path.name}")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> int:
    selected_videos = prompt_for_videos(list_raw_videos(RAW_VIDEO_DIR))

    for video_path in selected_videos:
        print()
        print(f"=== Running exploratory steps for {video_path.name} ===")
        for step in STEPS:
            run_step(step, video_path)

    print()
    print("Completed exploratory steps 01 to 04.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
