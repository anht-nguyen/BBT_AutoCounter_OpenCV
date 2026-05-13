from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .crossing import CrossingCounter, CrossingResult
from .hand_confirmation import HandFrameResult, HandMotionConfirmator
from .motion import ContourInfo, MotionMaskCleanerConfig, clean_motion_mask, draw_contour_debug, filter_contours_by_area
from .object_confirmation import ObjectTransferConfirmator, ObjectTransferResult
from .ui import add_panel_label, draw_text_block, mask_to_bgr, stack_views


@dataclass(frozen=True)
class PipelineConfig:
    partition_x: int
    direction: str = "left_to_right"
    crossing_zone_width: int = 100
    dead_zone_width: int = 20
    min_area: int = 500
    cooldown_frames: int = 20
    confirmation_mode: str = "hybrid"
    leading_edge_margin: int = 10
    target_confirmation_window_frames: int = 10
    target_motion_area_threshold: int = 300
    fingertip_margin: int = 5
    hand_mask_padding: int = 20
    hand_mask_dilation: int = 15
    target_non_hand_motion_threshold: int = 300
    persistence_motion_threshold: int = 120
    persistence_frames_required: int = 2
    absence_reset_frames: int = 4
    selected_fingertips: tuple[str, ...] = ("thumb", "index", "middle")
    fingertip_confirm_window_frames: int = 10
    object_confirm_window_frames: int = 5
    model_asset_path: str | None = None
    background_history: int = 500
    background_var_threshold: float = 50.0
    detect_shadows: bool = False
    motion_cleaner: MotionMaskCleanerConfig = MotionMaskCleanerConfig()


@dataclass
class ScoringSummary:
    combined_count: int = 0
    total_motion_events: int = 0
    total_hand_confirmed_events: int = 0
    rejected_motion_events_without_hand: int = 0
    rejected_motion_events_without_object: int = 0
    last_fingertip_cross_frame: int | None = None
    last_object_confirm_frame: int | None = None


@dataclass
class PipelineArtifacts:
    cleaned_mask: np.ndarray
    contour_info_list: list[ContourInfo]
    contour_debug_frame: np.ndarray
    motion_result: CrossingResult
    hand_result: HandFrameResult
    object_result: ObjectTransferResult
    scoring_frame: np.ndarray
    comparison_view: np.ndarray
    motion_event_detected: bool
    fingertip_recent: bool
    object_recent: bool
    hand_confirmed_score: bool


class BBTScoringPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.counter = CrossingCounter(
            partition_x=config.partition_x,
            direction=config.direction,
            crossing_zone_width=config.crossing_zone_width,
            dead_zone_width=config.dead_zone_width,
            min_area=config.min_area,
            cooldown_frames=config.cooldown_frames,
            leading_edge_margin=config.leading_edge_margin,
            confirmation_mode=config.confirmation_mode,
            target_confirmation_window_frames=config.target_confirmation_window_frames,
            target_motion_area_threshold=config.target_motion_area_threshold,
        )
        self.confirmator = HandMotionConfirmator(
            partition_x=config.partition_x,
            direction=config.direction,
            fingertip_margin=config.fingertip_margin,
            hand_mask_padding=config.hand_mask_padding,
            hand_mask_dilation=config.hand_mask_dilation,
            selected_fingertips=config.selected_fingertips,
            model_asset_path=config.model_asset_path,
        )
        self.object_confirmator = ObjectTransferConfirmator(
            partition_x=config.partition_x,
            direction=config.direction,
            target_motion_threshold=config.target_non_hand_motion_threshold,
            persistence_motion_threshold=config.persistence_motion_threshold,
            persistence_frames_required=config.persistence_frames_required,
            absence_reset_frames=config.absence_reset_frames,
        )
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.background_history,
            varThreshold=config.background_var_threshold,
            detectShadows=config.detect_shadows,
        )
        self.summary = ScoringSummary()

    def process_frame(self, frame: np.ndarray, frame_index: int) -> PipelineArtifacts:
        raw_mask = self.background_subtractor.apply(frame)
        cleaned_mask = clean_motion_mask(raw_mask, config=self.config.motion_cleaner)
        contour_info_list = filter_contours_by_area(cleaned_mask, min_area=self.config.min_area)
        contour_debug_frame = draw_contour_debug(frame, contour_info_list)

        motion_result = self.counter.update(cleaned_mask, frame_idx=frame_index)
        hand_result = self.confirmator.analyze_frame(frame, cleaned_mask)
        object_result = self.object_confirmator.analyze_frame(hand_result.non_hand_motion_mask)

        if hand_result.fingertip_crossed:
            self.summary.last_fingertip_cross_frame = frame_index
        if object_result.object_confirmed:
            self.summary.last_object_confirm_frame = frame_index

        fingertip_recent = self.summary.last_fingertip_cross_frame is not None and (frame_index - self.summary.last_fingertip_cross_frame) <= self.config.fingertip_confirm_window_frames
        object_recent = self.summary.last_object_confirm_frame is not None and (frame_index - self.summary.last_object_confirm_frame) <= self.config.object_confirm_window_frames

        motion_event_detected = motion_result.event_detected
        hand_confirmed_score = motion_event_detected and fingertip_recent and object_recent
        if motion_event_detected:
            self.summary.total_motion_events += 1
            if hand_confirmed_score:
                self.summary.combined_count += 1
                self.summary.total_hand_confirmed_events += 1
            else:
                if not fingertip_recent:
                    self.summary.rejected_motion_events_without_hand += 1
                if not object_recent:
                    self.summary.rejected_motion_events_without_object += 1

        scoring_frame = self.confirmator.draw_debug_overlay(frame, hand_result, target_non_hand_motion_threshold=self.config.target_non_hand_motion_threshold)
        scoring_frame = self.object_confirmator.draw_debug_overlay(scoring_frame, object_result)
        scoring_frame = self.counter.draw_debug_overlay(scoring_frame, motion_result)
        self._draw_scoring_text(scoring_frame, frame.shape[0], hand_result, object_result, motion_result, motion_event_detected, fingertip_recent, object_recent, hand_confirmed_score)
        cv2.putText(contour_debug_frame, f"Frame: {frame_index} | Contours kept: {len(contour_info_list)}", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        comparison_view = stack_views(
            [
                [add_panel_label(frame, "Original Frame"), add_panel_label(mask_to_bgr(cleaned_mask), "Cleaned Motion Mask")],
                [add_panel_label(contour_debug_frame, "Contour Debug"), add_panel_label(scoring_frame, "Scoring And Count")],
            ]
        )
        return PipelineArtifacts(
            cleaned_mask=cleaned_mask,
            contour_info_list=contour_info_list,
            contour_debug_frame=contour_debug_frame,
            motion_result=motion_result,
            hand_result=hand_result,
            object_result=object_result,
            scoring_frame=scoring_frame,
            comparison_view=comparison_view,
            motion_event_detected=motion_event_detected,
            fingertip_recent=fingertip_recent,
            object_recent=object_recent,
            hand_confirmed_score=hand_confirmed_score,
        )

    def _draw_scoring_text(
        self,
        scoring_frame: np.ndarray,
        frame_height: int,
        hand_result: HandFrameResult,
        object_result: ObjectTransferResult,
        motion_result: CrossingResult,
        motion_event_detected: bool,
        fingertip_recent: bool,
        object_recent: bool,
        hand_confirmed_score: bool,
    ) -> None:
        lines = [
            f"Scored: {'YES' if hand_confirmed_score else 'NO'} | Count: {self.summary.combined_count}",
            f"Motion status: {motion_result.score_status} | Motion mode: {motion_result.confirmation_mode} | State: {motion_result.state}",
            f"Fingertip crossed: {'YES' if hand_result.fingertip_crossed else 'NO'} | Recent: {'YES' if fingertip_recent else 'NO'}",
            f"Object confirmed: {'YES' if object_result.object_confirmed else 'NO'} | Recent: {'YES' if object_recent else 'NO'}",
            f"Motion event: {'YES' if motion_event_detected else 'NO'} | Hand-confirmed events: {self.summary.total_hand_confirmed_events}",
            f"Rejected by hand gate: {self.summary.rejected_motion_events_without_hand} | Rejected by object gate: {self.summary.rejected_motion_events_without_object}",
        ]
        draw_text_block(scoring_frame, lines, origin=(20, frame_height - 170), font_scale=0.68, line_spacing=30)
        if hand_confirmed_score:
            draw_text_block(scoring_frame, ["HAND-CONFIRMED SCORE!"], origin=(20, 210), font_scale=1.0, color=(0, 255, 120), thickness=3)

    def close(self) -> None:
        self.confirmator.close()
