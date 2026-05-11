from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


Point = tuple[int, int]


@dataclass
class BBTEnvironment:
    image_path: str
    image_width: int
    image_height: int
    box_polygon: list[Point]
    partition_line: list[Point]
    start_polygon: list[Point]
    target_polygon: list[Point]
    crossing_zone_polygon: list[Point]
    crossing_zone_width: int
    testing_hand: str
    start_side: str


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image


def _as_int_points(points: np.ndarray) -> list[Point]:
    return [(int(round(x)), int(round(y))) for x, y in points]


def _clip_points_to_image(points: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    clipped = points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, image_width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, image_height - 1)
    return clipped


def testing_hand_to_start_side(testing_hand: str) -> str:
    hand = testing_hand.lower()
    if hand == "left":
        return "right"
    if hand == "right":
        return "left"
    raise ValueError("testing_hand must be 'left' or 'right'")


def _line_intersection(
    line_a: tuple[np.ndarray, np.ndarray],
    line_b: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    p1, p2 = line_a
    q1, q2 = line_b
    denominator = (p1[0] - p2[0]) * (q1[1] - q2[1]) - (p1[1] - p2[1]) * (q1[0] - q2[0])
    if abs(float(denominator)) < 1e-8:
        raise ValueError("Lines are parallel and do not define a valid compartment split.")

    determinant_a = p1[0] * p2[1] - p1[1] * p2[0]
    determinant_b = q1[0] * q2[1] - q1[1] * q2[0]
    x = (determinant_a * (q1[0] - q2[0]) - (p1[0] - p2[0]) * determinant_b) / denominator
    y = (determinant_a * (q1[1] - q2[1]) - (p1[1] - p2[1]) * determinant_b) / denominator
    return np.array([x, y], dtype=np.float32)


def build_side_polygons(
    box_polygon: list[Point],
    partition_line: list[Point],
    start_side: str = "left",
) -> tuple[list[Point], list[Point]]:
    if len(box_polygon) != 4:
        raise ValueError("box_polygon must contain 4 points: top-left, top-right, bottom-right, bottom-left")
    if len(partition_line) != 2:
        raise ValueError("partition_line must contain 2 points: top, bottom")

    top_left, top_right, bottom_right, bottom_left = box_polygon
    partition_top, partition_bottom = partition_line

    partition_top_edge_point = _line_intersection(
        (np.asarray(partition_line[0], dtype=np.float32), np.asarray(partition_line[1], dtype=np.float32)),
        (np.asarray(top_left, dtype=np.float32), np.asarray(top_right, dtype=np.float32)),
    )
    partition_bottom_edge_point = _line_intersection(
        (np.asarray(partition_line[0], dtype=np.float32), np.asarray(partition_line[1], dtype=np.float32)),
        (np.asarray(bottom_left, dtype=np.float32), np.asarray(bottom_right, dtype=np.float32)),
    )

    split_top = _as_int_points(np.asarray([partition_top_edge_point], dtype=np.float32))[0]
    split_bottom = _as_int_points(np.asarray([partition_bottom_edge_point], dtype=np.float32))[0]

    left_polygon = [top_left, split_top, split_bottom, bottom_left]
    right_polygon = [split_top, top_right, bottom_right, split_bottom]

    if start_side == "left":
        return left_polygon, right_polygon
    if start_side == "right":
        return right_polygon, left_polygon
    raise ValueError("start_side must be 'left' or 'right'")


def build_crossing_zone_polygon(
    partition_line: list[Point],
    crossing_zone_width: int,
    image_width: int,
    image_height: int,
) -> list[Point]:
    if len(partition_line) != 2:
        raise ValueError("partition_line must contain 2 points")
    if crossing_zone_width <= 0:
        raise ValueError("crossing_zone_width must be positive")
    if image_width <= 0:
        raise ValueError("image_width must be positive")
    if image_height <= 0:
        raise ValueError("image_height must be positive")

    line = np.asarray(partition_line, dtype=np.float32)
    direction = line[1] - line[0]
    length = float(np.linalg.norm(direction))
    if length == 0:
        raise ValueError("partition_line points must be different")

    top_y = 0.0
    bottom_y = float(image_height - 1)
    if abs(float(direction[1])) < 1e-8:
        raise ValueError("partition_line must not be horizontal")

    top_x = line[0][0] + ((top_y - line[0][1]) / direction[1]) * direction[0]
    bottom_x = line[0][0] + ((bottom_y - line[0][1]) / direction[1]) * direction[0]

    extended_line = np.array(
        [
            [top_x, top_y],
            [bottom_x, bottom_y],
        ],
        dtype=np.float32,
    )
    unit_direction = (extended_line[1] - extended_line[0]) / float(np.linalg.norm(extended_line[1] - extended_line[0]))
    normal = np.array([-unit_direction[1], unit_direction[0]], dtype=np.float32)
    offset = normal * (crossing_zone_width / 2.0)

    polygon = np.array(
        [
            extended_line[0] + offset,
            extended_line[1] + offset,
            extended_line[1] - offset,
            extended_line[0] - offset,
        ],
        dtype=np.float32,
    )
    polygon = _clip_points_to_image(polygon, image_width=image_width, image_height=image_height)
    return _as_int_points(polygon)


def polygon_to_mask(image_shape: tuple[int, int, int], polygon: list[Point]) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
    return mask


def render_environment(image: np.ndarray, environment: BBTEnvironment, alpha: float = 0.28) -> np.ndarray:
    preview = image.copy()
    overlay = image.copy()

    start_polygon = np.asarray(environment.start_polygon, dtype=np.int32)
    target_polygon = np.asarray(environment.target_polygon, dtype=np.int32)
    crossing_polygon = np.asarray(environment.crossing_zone_polygon, dtype=np.int32)
    partition_line = np.asarray(environment.partition_line, dtype=np.int32)
    box_polygon = np.asarray(environment.box_polygon, dtype=np.int32)

    cv2.fillPoly(overlay, [start_polygon], (70, 180, 70))
    cv2.fillPoly(overlay, [target_polygon], (40, 170, 240))
    cv2.fillPoly(overlay, [crossing_polygon], (255, 120, 20))
    cv2.addWeighted(overlay, alpha, preview, 1 - alpha, 0, preview)

    cv2.polylines(preview, [box_polygon], isClosed=True, color=(255, 255, 255), thickness=3)
    cv2.polylines(preview, [start_polygon], isClosed=True, color=(70, 180, 70), thickness=2)
    cv2.polylines(preview, [target_polygon], isClosed=True, color=(40, 170, 240), thickness=2)
    cv2.polylines(preview, [crossing_polygon], isClosed=True, color=(255, 120, 20), thickness=2)
    cv2.line(preview, tuple(partition_line[0]), tuple(partition_line[1]), (30, 30, 220), 3)

    _draw_label(preview, "Start side", environment.start_polygon[0], (70, 180, 70))
    _draw_label(preview, "Target side", environment.target_polygon[1], (40, 170, 240))
    _draw_label(preview, "Crossing zone", environment.crossing_zone_polygon[0], (255, 120, 20))

    return preview


def _draw_label(image: np.ndarray, text: str, anchor: Point, color: tuple[int, int, int]) -> None:
    x, y = anchor
    label_origin = (max(x + 10, 15), max(y + 25, 30))
    cv2.putText(image, text, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def save_environment(environment: BBTEnvironment, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(environment), indent=2), encoding="utf-8")
    return output_path


def save_environment_preview(image: np.ndarray, environment: BBTEnvironment, output_path: Path) -> Path:
    preview = render_environment(image, environment)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), preview)
    if not success:
        raise RuntimeError(f"Could not write preview image: {output_path}")
    return output_path


def save_region_masks(
    image: np.ndarray,
    environment: BBTEnvironment,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    regions = {
        "start_side": environment.start_polygon,
        "target_side": environment.target_polygon,
        "crossing_zone": environment.crossing_zone_polygon,
    }
    saved_paths: dict[str, Path] = {}
    for name, polygon in regions.items():
        mask = polygon_to_mask(image.shape, polygon)
        mask_path = output_dir / f"{name}_mask.png"
        success = cv2.imwrite(str(mask_path), mask)
        if not success:
            raise RuntimeError(f"Could not write mask image: {mask_path}")
        saved_paths[name] = mask_path

    return saved_paths


def annotate_environment(
    image_path: Path,
    crossing_zone_width: int = 90,
    testing_hand: str = "right",
    start_side: str | None = None,
    window_name: str = "Define BBT Environment",
) -> tuple[BBTEnvironment, np.ndarray]:
    image = load_image(image_path)
    selector = _EnvironmentSelector(image=image, window_name=window_name)
    box_polygon, partition_line = selector.run()

    resolved_start_side = start_side or testing_hand_to_start_side(testing_hand)

    start_polygon, target_polygon = build_side_polygons(
        box_polygon=box_polygon,
        partition_line=partition_line,
        start_side=resolved_start_side,
    )
    crossing_zone_polygon = build_crossing_zone_polygon(
        partition_line=partition_line,
        crossing_zone_width=crossing_zone_width,
        image_width=int(image.shape[1]),
        image_height=int(image.shape[0]),
    )

    environment = BBTEnvironment(
        image_path=str(image_path),
        image_width=int(image.shape[1]),
        image_height=int(image.shape[0]),
        box_polygon=box_polygon,
        partition_line=partition_line,
        start_polygon=start_polygon,
        target_polygon=target_polygon,
        crossing_zone_polygon=crossing_zone_polygon,
        crossing_zone_width=int(crossing_zone_width),
        testing_hand=testing_hand.lower(),
        start_side=resolved_start_side,
    )
    return environment, image


class _EnvironmentSelector:
    def __init__(self, image: np.ndarray, window_name: str) -> None:
        self.image = image
        self.window_name = window_name
        self.display_scale = self._compute_display_scale(image)
        self.display_image = cv2.resize(
            image,
            None,
            fx=self.display_scale,
            fy=self.display_scale,
            interpolation=cv2.INTER_AREA if self.display_scale < 1 else cv2.INTER_LINEAR,
        )
        self.box_points: list[Point] = []
        self.partition_points: list[Point] = []
        self.step = "box"
        self.cancelled = False

    def run(self) -> tuple[list[Point], list[Point]]:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        try:
            while True:
                canvas = self._draw_canvas()
                cv2.imshow(self.window_name, canvas)
                key = cv2.waitKey(20) & 0xFF

                if key == 27 or key == ord("q"):
                    self.cancelled = True
                    break
                if key == ord("u"):
                    self._undo()
                elif key == ord("r"):
                    self._reset_current_step()
                elif key in (13, 32):
                    if self.step == "box" and len(self.box_points) == 4:
                        self.step = "partition"
                    elif self.step == "partition" and len(self.partition_points) == 2:
                        break
        finally:
            cv2.destroyWindow(self.window_name)

        if self.cancelled:
            raise RuntimeError("Annotation cancelled by user.")
        if len(self.box_points) != 4 or len(self.partition_points) != 2:
            raise RuntimeError("Annotation incomplete.")

        return self.box_points, self.partition_points

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        scaled_point = (int(round(x / self.display_scale)), int(round(y / self.display_scale)))
        if self.step == "box" and len(self.box_points) < 4:
            self.box_points.append(scaled_point)
        elif self.step == "partition" and len(self.partition_points) < 2:
            self.partition_points.append(scaled_point)

    def _undo(self) -> None:
        if self.step == "partition" and self.partition_points:
            self.partition_points.pop()
            return
        if self.step == "partition" and not self.partition_points:
            self.step = "box"
        if self.box_points:
            self.box_points.pop()

    def _reset_current_step(self) -> None:
        if self.step == "box":
            self.box_points.clear()
        else:
            self.partition_points.clear()

    def _draw_canvas(self) -> np.ndarray:
        canvas = self.display_image.copy()
        box_points_scaled = self._scale_points(self.box_points)
        partition_points_scaled = self._scale_points(self.partition_points)

        if box_points_scaled:
            for point in box_points_scaled:
                cv2.circle(canvas, point, 6, (255, 255, 255), -1)
                cv2.circle(canvas, point, 3, (0, 0, 0), -1)
            if len(box_points_scaled) > 1:
                cv2.polylines(canvas, [np.asarray(box_points_scaled, dtype=np.int32)], False, (255, 255, 255), 2)
            if len(box_points_scaled) == 4:
                cv2.polylines(canvas, [np.asarray(box_points_scaled, dtype=np.int32)], True, (255, 255, 255), 3)

        if len(partition_points_scaled) == 1:
            cv2.circle(canvas, partition_points_scaled[0], 6, (30, 30, 220), -1)
        elif len(partition_points_scaled) == 2:
            cv2.line(canvas, partition_points_scaled[0], partition_points_scaled[1], (30, 30, 220), 3)
            cv2.circle(canvas, partition_points_scaled[0], 6, (30, 30, 220), -1)
            cv2.circle(canvas, partition_points_scaled[1], 6, (30, 30, 220), -1)

        self._draw_instructions(canvas)
        return canvas

    def _draw_instructions(self, canvas: np.ndarray) -> None:
        lines = [
            "Step 1: click 4 box corners in order: top-left, top-right, bottom-right, bottom-left.",
            "Step 2: press Enter, then click partition top and partition bottom.",
            "Keys: Enter/Space = next step or finish, u = undo, r = reset current step, q/Esc = cancel.",
            f"Current step: {'box corners' if self.step == 'box' else 'partition line'}",
            f"Points: box {len(self.box_points)}/4, partition {len(self.partition_points)}/2",
        ]

        y = 30
        for line in lines:
            cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            y += 28

    def _scale_points(self, points: list[Point]) -> list[Point]:
        return [
            (
                int(round(point[0] * self.display_scale)),
                int(round(point[1] * self.display_scale)),
            )
            for point in points
        ]

    @staticmethod
    def _compute_display_scale(image: np.ndarray, max_width: int = 1500, max_height: int = 900) -> float:
        height, width = image.shape[:2]
        return min(max_width / width, max_height / height, 1.0)
