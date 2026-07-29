"""Interactive full-frame pixel to Pan/Tilt LUT calibration.

Run from the project root on Raspberry Pi:
    python -m src.tests.pointing_servo_calibration

Controls:
    A/D: pan -/+
    W/S: tilt -/+
    [ / ]: decrease/increase adjustment step
    Enter: save the current target
    Q or Esc: quit
"""

import json
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VISION_DIR = PROJECT_ROOT / "src" / "vision"
for path in (PROJECT_ROOT, VISION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_runtime_config import (  # noqa: E402
    FRAME_H,
    FRAME_W,
    LATEST_FRAME_CAPTURE,
    MIRROR,
    SERVO_POINTING_CALIB_PATH,
)
from pi_runtime_controller import PiSmartLightController  # noqa: E402
from raspberry_pi_runtime import LatestFrameCamera, open_camera  # noqa: E402
from src.config import (  # noqa: E402
    PAN_MAX_DEG,
    PAN_MIN_DEG,
    TILT_MAX_DEG,
    TILT_MIN_DEG,
)
from src.core.led_controller import LEDController  # noqa: E402


def calibration_targets():
    cx = FRAME_W // 2
    cy = FRAME_H // 2
    pan_pixels = [0, FRAME_W // 4, cx, (FRAME_W * 3) // 4, FRAME_W - 1]
    tilt_pixels = [0, FRAME_H // 4, cy, (FRAME_H * 3) // 4, FRAME_H - 1]
    return (
        [("pan", pixel, (pixel, cy)) for pixel in pan_pixels]
        + [("tilt", pixel, (cx, pixel)) for pixel in tilt_pixels]
    )


def save_calibration(path, pan_points, tilt_points):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "frame_size": [FRAME_W, FRAME_H],
        "mode": "residual",
        "pan": pan_points,
        "tilt": tilt_points,
    }
    output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[Calibration] saved {output}", flush=True)


def main():
    controller = PiSmartLightController()
    if controller.servo is None:
        raise RuntimeError("ServoController is unavailable")

    led = LEDController()
    cap = open_camera()
    if LATEST_FRAME_CAPTURE:
        cap = LatestFrameCamera(cap)

    targets = calibration_targets()
    target_index = 0
    pan_points = []
    tilt_points = []
    pan_deg = float(controller.servo_pan_deg)
    tilt_deg = float(controller.servo_tilt_deg)
    step = 1.0
    prepared_target_index = -1

    controller.servo.resume()
    controller.servo.move_to(pan_deg, tilt_deg)
    led.spot_on(50)

    try:
        while target_index < len(targets):
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            if controller.undistorter is not None:
                frame = controller.undistorter.undistort(frame)
            if MIRROR:
                frame = cv2.flip(frame, 1)

            axis, pixel, (target_x, target_y) = targets[target_index]
            base_pan_deg, base_tilt_deg = controller.base_servo_angles_for_pixel(
                (target_x, target_y)
            )
            if prepared_target_index != target_index:
                pan_deg = max(PAN_MIN_DEG, min(PAN_MAX_DEG, base_pan_deg))
                tilt_deg = max(TILT_MIN_DEG, min(TILT_MAX_DEG, base_tilt_deg))
                controller.servo.move_to(pan_deg, tilt_deg)
                prepared_target_index = target_index
            cv2.drawMarker(
                frame,
                (target_x, target_y),
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=36,
                thickness=2,
            )
            lines = [
                f"target {target_index + 1}/{len(targets)} axis={axis} pixel={pixel}",
                f"pan={pan_deg:.1f} tilt={tilt_deg:.1f} step={step:.1f}",
                f"base pan={base_pan_deg:.1f} tilt={base_tilt_deg:.1f}",
                "A/D pan  W/S tilt  [/] step  Enter save  Q quit",
            ]
            for line_index, text in enumerate(lines):
                cv2.putText(
                    frame,
                    text,
                    (12, 28 + line_index * 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Servo Pointing Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            changed = False
            if key == ord("a"):
                pan_deg -= step
                changed = True
            elif key == ord("d"):
                pan_deg += step
                changed = True
            elif key == ord("w"):
                tilt_deg -= step
                changed = True
            elif key == ord("s"):
                tilt_deg += step
                changed = True
            elif key == ord("["):
                step = max(0.1, step / 2.0)
            elif key == ord("]"):
                step = min(10.0, step * 2.0)
            elif key in (10, 13):
                if not controller.servo.wait_until_done():
                    print(
                        "[Calibration] servo did not reach the target; retry",
                        flush=True,
                    )
                    continue
                actual_pan, actual_tilt = controller.servo.get_position()
                if axis == "pan":
                    correction_deg = actual_pan - base_pan_deg
                    pan_points.append(
                        {
                            "pixel": pixel,
                            "correction_deg": round(correction_deg, 3),
                        }
                    )
                else:
                    correction_deg = actual_tilt - base_tilt_deg
                    tilt_points.append(
                        {
                            "pixel": pixel,
                            "correction_deg": round(correction_deg, 3),
                        }
                    )
                print(
                    f"[Calibration] recorded {axis} pixel={pixel} "
                    f"correction={correction_deg:+.3f}deg",
                    flush=True,
                )
                target_index += 1
                continue
            elif key in (27, ord("q")):
                print("[Calibration] cancelled", flush=True)
                return

            if changed:
                pan_deg = max(PAN_MIN_DEG, min(PAN_MAX_DEG, pan_deg))
                tilt_deg = max(TILT_MIN_DEG, min(TILT_MAX_DEG, tilt_deg))
                controller.servo.move_to(pan_deg, tilt_deg)

        save_calibration(
            SERVO_POINTING_CALIB_PATH,
            pan_points,
            tilt_points,
        )
    finally:
        cap.release()
        led.shutdown()
        controller.servo.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
