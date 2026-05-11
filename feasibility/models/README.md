# MediaPipe Models

Place local MediaPipe Task models here when a script requires them.

For `feasibility/exploratory/05_hand_motion_confirmation.py`, the default lookup path is:

- `feasibility/models/hand_landmarker.task`

You can also override the path at runtime:

```bash
python feasibility/exploratory/05_hand_motion_confirmation.py --model-asset-path path/to/hand_landmarker.task
```

The `.task` model files are ignored by Git and are intended to stay local.
