# MediaPipe Models

Place local MediaPipe Task models here when a script requires them.

For the current MediaPipe-based exploratory scripts, the default lookup path is:

- `feasibility/models/hand_landmarker.task`

Scripts that use this model include:

- `feasibility/exploratory/05_hand_motion_confirmation.py`
- `feasibility/exploratory/06_scoring_w_hand_confirm.py`

You can also override the path at runtime:

```bash
python feasibility/exploratory/05_hand_motion_confirmation.py --model-asset-path path/to/hand_landmarker.task
```

```bash
python feasibility/exploratory/06_scoring_w_hand_confirm.py --model-asset-path path/to/hand_landmarker.task
```

The `.task` model files are ignored by Git and are intended to stay local.
