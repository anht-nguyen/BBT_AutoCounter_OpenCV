# Project Plan: Automated Box and Block Test Scoring Using Computer Vision

## Project Title

**Automated Box and Block Test Scoring Using Computer Vision**

## Project Summary

This project focuses on developing and evaluating a computer vision prototype for automated scoring of the Box and Block Test (BBT). The system will use a controlled setup with one camera, fixed box position, consistent lighting, and simple blocks. Development will begin with recorded videos to allow systematic testing and error analysis. After offline feasibility is established, the project will assess whether the same approach could support online automated scoring.

The primary goal is not perfect automatic scoring. Instead, the project aims to determine whether a simple, explainable computer vision pipeline can estimate BBT scores with acceptable accuracy under controlled recording conditions.

## Overall Objective

Develop and evaluate a prototype computer vision pipeline that estimates Box and Block Test scores from video recordings and assesses feasibility for future online automated scoring.

## Research Questions

1. Can a simple computer vision pipeline estimate Box and Block Test scores from video recordings with acceptable accuracy under controlled recording conditions?
2. Is the approach feasible for online automated scoring?
3. What recording conditions make automated scoring more or less reliable?
4. What are the most common sources of automated scoring error?
5. Can simple rule-based computer vision methods provide a practical starting point before using more complex machine learning or deep learning approaches?

## Project Scope

### Included in Scope

- Controlled video recording setup
- One-camera recording approach
- Fixed BBT box position
- Consistent lighting
- Clear view of the partition
- Simple or colored blocks if available
- 60-second BBT-style trials
- Offline video-based algorithm development
- Rule-based OpenCV counting algorithm
- Manual scoring for ground truth
- Automated vs manual score comparison
- Feasibility assessment for future online scoring

### Not Included in Initial Scope

- Perfect automatic scoring
- Clinical-grade validation
- Multi-site deployment
- Fully generalizable scoring across all camera angles and lighting conditions
- Training a deep learning model from scratch
- Robust scoring under severe occlusion
- Direct replacement of clinical manual scoring

## Specific Aims

## Aim 1: Build a Standardized Video Recording Setup

### Goal

Develop a controlled recording protocol for capturing BBT trials in a way that makes computer vision analysis feasible.

### Rationale

A standardized video setup reduces unnecessary sources of variation and makes the computer vision problem more manageable. The box, partition, start side, and target side should remain visible throughout the trial.

### Proposed Setup

- Camera positioned above or slightly in front of the BBT box
- Tripod or fixed camera mount
- Fixed box location
- Consistent lighting
- High-contrast background
- Colored or visually distinct blocks if available
- Tape markers for box boundary and partition line
- 60-second trials
- Clear view of the start compartment, target compartment, and partition

### Expected Output

A repeatable video recording protocol and a small set of recorded BBT-style videos suitable for algorithm development and evaluation.

## Aim 2: Develop a Computer Vision Counting Algorithm

### Goal

Build a simple OpenCV-based algorithm that estimates BBT score by detecting block transfer events.

### Primary Approach: Crossing-Line Detection

The algorithm will define a virtual line at or near the partition. A transfer event is counted when a block-like moving object crosses from the start side into the target side.

### Basic Pipeline

1. Load recorded video.
2. Define regions of interest:
   - Start compartment
   - Target compartment
   - Partition zone
3. Define a virtual crossing line at the partition.
4. Detect motion or block-like objects near the partition.
5. Determine whether the movement direction is from start side to target side.
6. Count one valid event per detected transfer.
7. Add rules to avoid double-counting.
8. Save the final automated score and optional annotated output video.

### Example Counting Rule

> Count one event when a moving block-like object crosses the partition line from the start compartment into the target compartment and disappears, settles, or is confirmed in the target region.

### Strengths of This Approach

- Easy to implement
- Explainable
- Good fit for a controlled setup
- Does not require a large labeled dataset
- Easier to debug than deep learning approaches
- Suitable for rapid feasibility testing

### Known Limitations

- May fail when the hand occludes the block
- May double-count if the same block moves back and forth near the line
- May undercount when multiple blocks are moved together
- May be sensitive to lighting and camera angle
- May struggle if the block and background have low contrast

### Expected Output

A working prototype that processes recorded videos and generates automated BBT score estimates.

## Aim 3: Validate Automated Count Against Manual Count

### Goal

Evaluate how closely the automated scoring system matches manual scoring.

### Validation Strategy

Manual scoring will serve as the ground truth. Each video will be manually reviewed and assigned a BBT count. The automated count will then be compared against the manual score.

### Suggested Metrics

- Manual count
- Automated count
- Absolute error
- Percent error
- Mean absolute error
- Correlation with manual count
- Number of overcounts
- Number of undercounts
- Qualitative failure-case notes

### Failure Cases to Track

- Hand covering the block
- Two blocks moved together
- Dropped block
- Poor lighting
- Camera shift
- Low contrast between block and background
- Block pile-up in the target compartment
- Motion detected from the hand rather than the block

### Expected Output

A feasibility analysis showing how well the prototype estimates manual BBT scores and where the algorithm fails.

## Deliverables

## Technical Deliverables

1. Video recording protocol
2. Small recorded video dataset
3. Manual scoring spreadsheet
4. Python/OpenCV prototype
5. Region-of-interest selection method
6. Crossing-line detection algorithm
7. Automated count output
8. Annotated example videos
9. Evaluation script or notebook
10. Error analysis summary

## Documentation Deliverables

1. Repository README
2. Project plan
3. Recording setup guide
4. Algorithm description
5. Results summary
6. Limitations and future work section

## Final Presentation Deliverables

1. Short slide deck
2. Demo video or live algorithm demonstration
3. Manual vs automated scoring results
4. Failure-case examples
5. Future directions for online scoring

## Tentative 6-Week Timeline

## Week 1: Background, Setup, and Pilot Recording

### Goals

- Understand the Box and Block Test
- Define the project scope
- Set up the development environment
- Create a controlled recording setup
- Record pilot videos

### Tasks

- Review BBT scoring rules and standard administration
- Install Python, OpenCV, NumPy, pandas, and plotting tools
- Set up GitHub repository
- Determine camera position
- Mark the box boundary and partition line
- Record several pilot videos
- Manually score pilot videos

### Deliverables

- Initial project scope
- Recording setup draft
- Pilot videos
- Manual scoring sheet template
- Working Python environment

## Week 2: Dataset Collection and Ground-Truth Scoring

### Goals

- Collect a small video dataset
- Create manual ground-truth scores
- Identify visual challenges before algorithm development

### Tasks

- Record BBT-style trials under controlled conditions
- Save videos using consistent file naming
- Manually score each video
- Record notes about lighting, camera angle, and visible errors
- Organize video files and scoring table

### Deliverables

- Initial video dataset
- Manual scoring spreadsheet
- Data organization structure
- Notes on recording quality and common challenges

## Week 3: First Computer Vision Prototype

### Goals

- Build the first version of the video-processing pipeline
- Define regions of interest
- Detect motion near the partition

### Tasks

- Load video frame by frame
- Crop or define the BBT region of interest
- Draw partition line and compartment boundaries
- Apply simple preprocessing
- Test background subtraction or frame differencing
- Detect movement near the partition
- Generate an annotated output video

### Deliverables

- First prototype script
- Region-of-interest definition method
- Annotated pilot video
- Initial observations on what works and what fails

## Week 4: Crossing-Line Counting Algorithm

### Goals

- Implement transfer-event counting
- Reduce false positives and double-counting
- Test the algorithm on multiple videos

### Tasks

- Add crossing-line logic
- Track movement direction
- Add event cooldown to avoid double-counting
- Filter detected objects by size, location, or motion direction
- Test the algorithm on the video dataset
- Compare early automated counts with manual counts

### Deliverables

- Working crossing-line counting algorithm
- Automated counts for multiple videos
- Preliminary manual vs automated comparison
- List of common algorithm errors

## Week 5: Evaluation and Error Analysis

### Goals

- Evaluate automated scoring accuracy
- Analyze failure cases
- Assess feasibility for online scoring

### Tasks

- Run the algorithm on all recorded videos
- Calculate absolute error and percent error
- Calculate mean absolute error
- Plot manual vs automated scores
- Identify overcount and undercount examples
- Summarize failure cases
- Discuss what would be needed for live scoring

### Deliverables

- Final results table
- Evaluation figures
- Failure-case examples
- Feasibility summary for online scoring

## Week 6: Final Documentation and Presentation

### Goals

- Clean the repository
- Finalize documentation
- Prepare final presentation and demo

### Tasks

- Refactor code
- Add comments and usage instructions
- Finalize README
- Write final project summary
- Prepare slides
- Select example videos for demonstration
- Summarize limitations and future directions

### Deliverables

- Final GitHub repository
- Final README
- Final project report or summary document
- Final slide deck
- Demo video or live demonstration
- Future work recommendations

## Suggested Success Criteria

### Minimum Success

- The program can load recorded BBT videos.
- The user can define the relevant region of interest.
- The program outputs an automated count.
- Manual and automated counts can be compared.

### Good Success

- The system produces reasonable counts under controlled conditions.
- The average error is quantified.
- The main sources of error are clearly documented.
- The project includes an annotated video output.

### Excellent Success

- The system achieves relatively low counting error under controlled recording conditions.
- The pipeline is organized, documented, and reproducible.
- The project provides a clear feasibility assessment for future online scoring.

## Recommended Final Framing

This project should be framed as a feasibility study:

> This project evaluates whether a simple, explainable computer vision pipeline can estimate Box and Block Test scores from controlled video recordings and whether the approach could be extended toward online automated scoring.

The final interpretation should emphasize feasibility, limitations, and next steps rather than claiming that the system replaces manual clinical scoring.
