# Automated Box and Block Test Scoring Using Computer Vision

## Overview

This repository supports the development and evaluation of a computer vision prototype for automated scoring of the Box and Block Test (BBT). The project focuses on estimating the number of successfully transferred blocks from video recordings using a controlled camera setup and an explainable computer vision pipeline.

The goal is not to create a perfect or clinically deployable scoring system at this stage. Instead, the project aims to test whether a simple, transparent computer vision approach can approximate manual BBT scoring under controlled recording conditions and provide a foundation for future online or real-time scoring.

## Project Scope

The prototype is designed for a controlled setup with:

- One camera
- Fixed box position
- Consistent lighting
- Clear view of the partition
- Simple, visually distinguishable blocks
- Standardized 60-second BBT trials
- Recorded videos used for development, testing, and feasibility evaluation

The initial development phase uses pre-recorded videos to build and validate the algorithm. Once video-based feasibility is established, the same logic may be adapted for online scoring settings.

## Research Goal

To develop and evaluate a computer vision pipeline that estimates Box and Block Test scores from video recordings under controlled recording conditions.

## Core Research Questions

1. Can a simple computer vision pipeline estimate BBT scores from video recordings with acceptable accuracy under controlled recording conditions?
2. Is the proposed approach feasible for online automated scoring?
3. What are the main sources of counting error, such as hand occlusion, poor lighting, dropped blocks, or multiple blocks moved together?

## Technical Approach

The initial approach uses crossing-line detection.

A virtual line is defined at or near the partition between the start and target compartments. The algorithm detects block-like motion near this line and counts one valid transfer event when a moving object crosses from the start side into the target side.

This approach was selected because it is:

- Simple
- Explainable
- Suitable for controlled setups
- Compatible with OpenCV-based development
- Easier to debug than deep learning methods
- Appropriate for feasibility testing

## Proposed Computer Vision Pipeline

The expected workflow is:

1. Load recorded BBT video.
2. Define regions of interest:
   - Start compartment
   - Target compartment
   - Partition zone
   - Crossing line
3. Detect motion or block-like objects near the partition.
4. Track movement direction across the partition.
5. Count a valid transfer event when a block-like object crosses from the start side to the target side.
6. Apply rules to reduce double-counting.
7. Export the automated count.
8. Compare automated scoring against manual scoring.

## Example Counting Rule

A simple starting rule may be:

> Count one event when a moving block-like object crosses the partition line from the start compartment into the target compartment and disappears, settles, or is confirmed in the target region.

This rule can later be refined using object size filtering, motion direction, event cooldown periods, and target-region confirmation.

## Evaluation Plan

Automated scores will be compared with manual scores from recorded videos.

Suggested evaluation metrics include:

- Manual count
- Automated count
- Absolute error
- Percent error
- Mean absolute error
- Correlation between manual and automated scores
- Number of overcounts
- Number of undercounts
- Qualitative failure-case analysis

Common failure cases to document include:

- Hand occluding the block
- Two blocks moved at once
- Dropped blocks
- Poor lighting
- Camera movement
- Low contrast between blocks and background
- Block pile-up in the target compartment

## Expected Outputs

This repository may include:

- Video processing scripts
- Region-of-interest selection tools
- Crossing-line detection algorithm
- Annotated output videos
- Manual vs automated scoring tables
- Error analysis notebooks
- Example figures
- Documentation for recording setup and algorithm use

## Repository Structure

A possible repository structure is:

```text
.
├── README.md
├── data/
│   ├── raw_videos/
│   ├── annotated_videos/
│   └── scoring_tables/
├── src/
│   ├── preprocess_video.py
│   ├── define_roi.py
│   ├── count_crossings.py
│   └── evaluate_counts.py
├── notebooks/
│   └── feasibility_analysis.ipynb
├── results/
│   ├── figures/
│   ├── tables/
│   └── example_outputs/
└── docs/
    ├── recording_protocol.md
    └── project_plan_6_week.md
```

## Development Notes

This project should begin with recorded video rather than live scoring. Offline video analysis allows easier debugging, repeatable testing, manual comparison, and systematic error analysis.

After the offline pipeline is working, the same logic can be adapted to an online setting using a live webcam feed.

## Limitations

This prototype is expected to work best in controlled conditions. It may not generalize to all clinical environments, camera angles, lighting conditions, participant movement patterns, or box/block designs.

The primary goal is feasibility, not perfect automatic scoring.

## Future Directions

Potential future improvements include:

- Online webcam-based scoring
- More robust object detection
- Multi-camera setup
- Depth camera integration
- Automatic camera calibration
- Improved handling of occlusion
- Integration with clinical data collection workflows
- Validation in broader participant populations
