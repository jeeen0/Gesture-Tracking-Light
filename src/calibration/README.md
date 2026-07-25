# Pointing servo calibration

Run the interactive calibration after ROI, depth, and ray-mode behavior have
been verified on Raspberry Pi:

```bash
python -m src.tests.pointing_servo_calibration
```

Controls:

- `A` / `D`: pan - / +
- `W` / `S`: tilt - / +
- `[` / `]`: decrease / increase the adjustment step
- `Enter`: record the current target
- `Q` or `Esc`: cancel

The completed calibration is saved to
`src/calibration/servo_pointing_calibration.json`. The normal runtime loads
that file automatically. If the file is absent or invalid, the existing
center/gain/offset calculation is used.

Useful runtime switches:

```bash
PI_ROI_INPUT_SIZE=320
PI_LATEST_FRAME_CAPTURE=1
PI_DEPTH_ASYNC=1
PI_DEPTH_ASYNC_FPS=4
PI_DEPTH_RESULT_MAX_AGE=0.75
PI_TORCH_NUM_THREADS=2
PI_POINT_RAY_MODE=mcp_tip
PI_POINT_ENTRY_GRACE_SECONDS=1.0
PI_POINT_ARM_WINDOW=5
PI_POINT_ARM_MIN_HITS=3
```

Use `PI_POINT_RAY_MODE=finger_axis` only for an A/B comparison when the
on-screen target marker is consistently inaccurate.
