from __future__ import annotations

from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEASIBILITY_ROOT = PROJECT_ROOT / "feasibility"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feasibility.src.define_BBT_environment import annotate_environment
from feasibility.src.define_BBT_environment import save_environment
from feasibility.src.define_BBT_environment import save_environment_preview
from feasibility.src.define_BBT_environment import save_region_masks


IMAGE_PATH = FEASIBILITY_ROOT / "data" / "images" / "BBT-setup-image.jpg"
OUTPUT_DIR = FEASIBILITY_ROOT / "data" / "images" / "annotations"
OUTPUT_JSON = OUTPUT_DIR / "BBT_environment.json"
OUTPUT_PREVIEW = OUTPUT_DIR / "BBT_environment_preview.jpg"
OUTPUT_MASK_DIR = OUTPUT_DIR / "masks"
CROSSING_ZONE_WIDTH = 90
START_SIDE = "left"


def main() -> int:
    environment, image = annotate_environment(
        image_path=IMAGE_PATH,
        crossing_zone_width=CROSSING_ZONE_WIDTH,
        start_side=START_SIDE,
    )

    save_environment(environment, OUTPUT_JSON)
    save_environment_preview(image, environment, OUTPUT_PREVIEW)
    mask_paths = save_region_masks(image, environment, OUTPUT_MASK_DIR)

    print(f"Saved environment JSON to: {OUTPUT_JSON}")
    print(f"Saved preview image to: {OUTPUT_PREVIEW}")
    for name, path in mask_paths.items():
        print(f"Saved {name} mask to: {path}")
    print("Press any key in the preview window to close.")

    preview = cv2.imread(str(OUTPUT_PREVIEW))
    cv2.imshow("BBT Environment Preview", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
