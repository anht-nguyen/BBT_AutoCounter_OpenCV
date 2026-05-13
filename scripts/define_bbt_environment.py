from __future__ import annotations

from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bbt_autocounter.environment import annotate_environment, save_environment, save_environment_preview, save_region_masks


FEASIBILITY_ROOT = PROJECT_ROOT / "feasibility"
IMAGE_PATH = FEASIBILITY_ROOT / "data" / "images" / "BBT-setup-image.jpg"
OUTPUT_DIR = FEASIBILITY_ROOT / "data" / "images" / "annotations"
OUTPUT_JSON = OUTPUT_DIR / "BBT_environment.json"
OUTPUT_PREVIEW = OUTPUT_DIR / "BBT_environment_preview.jpg"
OUTPUT_MASK_DIR = OUTPUT_DIR / "masks"
CROSSING_ZONE_WIDTH = 90
START_SIDE = "left"


def main() -> int:
    environment, image = annotate_environment(image_path=IMAGE_PATH, crossing_zone_width=CROSSING_ZONE_WIDTH, start_side=START_SIDE)
    save_environment(environment, OUTPUT_JSON)
    save_environment_preview(image, environment, OUTPUT_PREVIEW)
    mask_paths = save_region_masks(image, environment, OUTPUT_MASK_DIR)
    print(f"Saved environment JSON to: {OUTPUT_JSON}")
    print(f"Saved preview image to: {OUTPUT_PREVIEW}")
    for name, path in mask_paths.items():
        print(f"Saved {name} mask to: {path}")
    preview = cv2.imread(str(OUTPUT_PREVIEW))
    if preview is None:
        raise FileNotFoundError(f"Could not load preview image: {OUTPUT_PREVIEW}")
    cv2.imshow("BBT Environment Preview", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
