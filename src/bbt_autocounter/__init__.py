from .crossing import CrossingCounter, CrossingResult
from .environment import (
    BBTEnvironment,
    annotate_environment,
    build_crossing_zone_polygon,
    build_side_polygons,
    load_environment,
    polygon_to_mask,
    render_environment,
    save_environment,
    save_environment_preview,
    save_region_masks,
)
from .hand_confirmation import HandFrameResult, HandMotionConfirmator
from .motion import (
    ContourInfo,
    CrossingZoneMotionDetector,
    MotionBlob,
    MotionDetectionResult,
    MotionDetectorConfig,
    MotionMaskCleanerConfig,
    clean_motion_mask,
    draw_contour_debug,
    filter_contours_by_area,
)
from .object_confirmation import ObjectTransferConfirmator, ObjectTransferResult
from .pipeline import BBTScoringPipeline, PipelineArtifacts, PipelineConfig, ScoringSummary

__all__ = [
    "BBTEnvironment",
    "BBTScoringPipeline",
    "ContourInfo",
    "CrossingCounter",
    "CrossingResult",
    "CrossingZoneMotionDetector",
    "HandFrameResult",
    "HandMotionConfirmator",
    "MotionBlob",
    "MotionDetectionResult",
    "MotionDetectorConfig",
    "MotionMaskCleanerConfig",
    "ObjectTransferConfirmator",
    "ObjectTransferResult",
    "PipelineArtifacts",
    "PipelineConfig",
    "ScoringSummary",
    "annotate_environment",
    "build_crossing_zone_polygon",
    "build_side_polygons",
    "clean_motion_mask",
    "draw_contour_debug",
    "filter_contours_by_area",
    "load_environment",
    "polygon_to_mask",
    "render_environment",
    "save_environment",
    "save_environment_preview",
    "save_region_masks",
]
