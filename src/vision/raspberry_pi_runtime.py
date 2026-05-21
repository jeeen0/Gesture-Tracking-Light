import atexit
import logging
import sys
import shutil
import subprocess
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from pi_runtime_config import (
    CAM_INDEX,
    CAMERA_BACKEND,
    DEBUG_OUTPUT,
    ENABLE_GESTURE_ZONE_ROI,
    ENABLE_HAND_SIZE_GATING,
    ENABLE_MOTION_ROI,
    ENABLE_YOLO,
    FRAME_H,
    FRAME_W,
    GESTURE_ZONE,
    MIN_FINE_GESTURE_HAND_BBOX_WIDTH_PX,
    MIN_MOTION_ROI_SIZE,
    MIRROR,
    MOTION_ROI_PADDING_RATIO,
    MP_DET_CONF,
    MP_TRK_CONF,
    MP_MODEL_COMPLEXITY,
    ACTIVE_INFERENCE_FPS,
    STANDBY_INFERENCE_FPS,
    POINT_INFERENCE_FPS,
    TRACK_ROI_SCALE,
    TRACK_ROI_PADDING_RATIO,
    TRACK_ROI_TTL,
    FULL_FRAME_REACQUIRE_INTERVAL,
    YOLO_AFTER_MISSES,
    YOLO_REACQUIRE_INTERVAL,
    ROI_FALLBACK_AFTER_MISSES,
    ROI_FALLBACK_INTERVAL,
    ROI_SCALE,
    SHOW_PREVIEW,
    STATUS_INTERVAL,
    TARGET_FPS,
    WAVE_CANDIDATE_FRAMES,
    WAVE_CONFIRM_MIN_HITS,
    WAVE_CONFIRM_WINDOW_SECONDS,
    WAVE_MOTION_SPAN_RATIO,
    YOLO_CONFIDENCE,
    YOLO_ROI_PADDING_RATIO,
    YOLO_ROI_SCALE,
)
from pi_runtime_config import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pi_runtime_controller import PiSmartLightController
from pi_runtime_events import emit_state
from src.core.led_controller import LEDController


class OpenCVCamera:
    def __init__(self, cap, name):
        self.cap = cap
        self.name = name

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class Picamera2Camera:
    def __init__(self, picam2):
        self.picam2 = picam2
        self.name = "picamera2"

    def read(self):
        try:
            frame = self.picam2.capture_array()
        except Exception as e:
            print(f"[PI] picamera2 capture failed: {e}", flush=True)
            return False, None
        if frame is None:
            return False, None
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return True, frame

    def release(self):
        self.picam2.stop()


class RpicamVidCamera:
    def __init__(self, proc, command_name):
        self.proc = proc
        self.command_name = command_name
        self.name = command_name
        self.buffer = bytearray()

    def read(self):
        deadline = time.time() + 2.0
        while time.time() < deadline:
            start = self.buffer.find(b"\xff\xd8")
            end = self.buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
            if start >= 0 and end >= 0:
                jpeg = bytes(self.buffer[start:end + 2])
                del self.buffer[:end + 2]
                frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame

            chunk = self.proc.stdout.read(4096)
            if not chunk:
                return False, None
            self.buffer.extend(chunk)
            if len(self.buffer) > 5_000_000:
                del self.buffer[:-1_000_000]
        return False, None

    def release(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def open_camera():
    backend = CAMERA_BACKEND
    errors = []
    if backend not in {"auto", "opencv", "picamera2", "rpicam-vid", "libcamera-vid"}:
        print(f"[PI] unknown PI_CAMERA_BACKEND={backend!r}; using auto", flush=True)
        backend = "auto"

    if backend in {"auto", "opencv"}:
        opencv_camera = open_opencv_camera(errors)
        if opencv_camera is not None:
            return opencv_camera

    if backend in {"auto", "picamera2"}:
        picamera = open_picamera2(errors)
        if picamera is not None:
            return picamera

    if backend in {"auto", "rpicam-vid", "libcamera-vid"}:
        rpicam = open_rpicam_vid(errors, backend)
        if rpicam is not None:
            return rpicam

    print("[PI] camera open failed", flush=True)
    for error in errors:
        print(f"[PI]   {error}", flush=True)
    print(
        "[PI] Try: rpicam-hello --list-cameras, ls -l /dev/video*, "
        "or PI_CAMERA_BACKEND=rpicam-vid python3 src/vision/raspberry_pi_runtime.py",
        flush=True,
    )
    raise RuntimeError("No camera backend produced frames")


def open_opencv_camera(errors):
    if isinstance(CAM_INDEX, int):
        api_candidates = [
            (cv2.CAP_V4L2, "opencv_v4l2"),
            (cv2.CAP_ANY, "opencv_any"),
        ]
    else:
        api_candidates = [(cv2.CAP_ANY, "opencv_any")]

    for api, name in api_candidates:
        cap = cv2.VideoCapture(CAM_INDEX, api)
        configure_opencv_capture(cap)
        ok, frame = read_warmup_frame(cap)
        if cap.isOpened() and ok and frame is not None:
            print(f"[PI] camera opened with {name} source={CAM_INDEX}", flush=True)
            return OpenCVCamera(cap, name)
        errors.append(f"{name} source={CAM_INDEX}: opened={cap.isOpened()} first_frame={ok}")
        cap.release()
    return None


def configure_opencv_capture(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def read_warmup_frame(cap):
    ok = False
    frame = None
    for _ in range(10):
        ok, frame = cap.read()
        if ok and frame is not None:
            return ok, frame
        time.sleep(0.05)
    return ok, frame


def open_picamera2(errors):
    try:
        from picamera2 import Picamera2
    except Exception as e:
        dist_packages = "/usr/lib/python3/dist-packages"
        if dist_packages not in sys.path:
            sys.path.append(dist_packages)
            try:
                from picamera2 import Picamera2
            except Exception as retry_error:
                errors.append(f"picamera2 import failed: {e}; retry failed: {retry_error}")
                return None
        else:
            errors.append(f"picamera2 import failed: {e}")
            return None

    picam2 = None
    try:
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},
            controls={"FrameRate": TARGET_FPS},
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(0.3)
        camera = Picamera2Camera(picam2)
        ok, frame = camera.read()
        if ok and frame is not None:
            print(f"[PI] camera opened with picamera2 source={CAM_INDEX}", flush=True)
            return camera
        errors.append("picamera2: first_frame=False")
    except Exception as e:
        errors.append(f"picamera2 open failed: {e}")

    if picam2 is not None:
        try:
            picam2.stop()
        except Exception:
            pass
    return None


def open_rpicam_vid(errors, backend):
    command_names = []
    if backend == "libcamera-vid":
        command_names.append("libcamera-vid")
    else:
        command_names.append("rpicam-vid")
    command_names.extend(name for name in ("rpicam-vid", "libcamera-vid") if name not in command_names)

    for command_name in command_names:
        command_path = shutil.which(command_name)
        if command_path is None:
            errors.append(f"{command_name}: command not found")
            continue

        cmd = [
            command_path,
            "--timeout",
            "0",
            "--nopreview",
            "--codec",
            "mjpeg",
            "--width",
            str(FRAME_W),
            "--height",
            str(FRAME_H),
            "--framerate",
            str(TARGET_FPS),
            "--output",
            "-",
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except Exception as e:
            errors.append(f"{command_name} start failed: {e}")
            continue

        camera = RpicamVidCamera(proc, command_name)
        ok, frame = camera.read()
        if ok and frame is not None:
            print(f"[PI] camera opened with {command_name} mjpeg pipe", flush=True)
            return camera

        return_code = proc.poll()
        camera.release()
        errors.append(f"{command_name}: first_frame=False returncode={return_code}")

    return None


def has_hand(results):
    return bool(results and getattr(results, "multi_hand_landmarks", None))


def clamp_roi_box(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(frame_width - 1, int(x1)))
    y1 = max(0, min(frame_height - 1, int(y1)))
    x2 = max(x1 + 1, min(frame_width, int(x2)))
    y2 = max(y1 + 1, min(frame_height, int(y2)))
    return x1, y1, x2, y2


def gesture_zone_box(frame_width, frame_height):
    return clamp_roi_box(
        (
            frame_width * GESTURE_ZONE["x_min_ratio"],
            frame_height * GESTURE_ZONE["y_min_ratio"],
            frame_width * GESTURE_ZONE["x_max_ratio"],
            frame_height * GESTURE_ZONE["y_max_ratio"],
        ),
        frame_width,
        frame_height,
    )


def motion_roi_box_from_thresh(thresh, frame_width, frame_height):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return None

    small_height, small_width = thresh.shape
    scale_x = frame_width / max(small_width, 1)
    scale_y = frame_height / max(small_height, 1)

    x1 = x * scale_x
    y1 = y * scale_y
    x2 = (x + w) * scale_x
    y2 = (y + h) * scale_y

    pad = max(x2 - x1, y2 - y1) * MOTION_ROI_PADDING_RATIO
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad

    if (x2 - x1) < MIN_MOTION_ROI_SIZE:
        center_x = (x1 + x2) / 2
        x1 = center_x - MIN_MOTION_ROI_SIZE / 2
        x2 = center_x + MIN_MOTION_ROI_SIZE / 2

    if (y2 - y1) < MIN_MOTION_ROI_SIZE:
        center_y = (y1 + y2) / 2
        y1 = center_y - MIN_MOTION_ROI_SIZE / 2
        y2 = center_y + MIN_MOTION_ROI_SIZE / 2

    return clamp_roi_box((x1, y1, x2, y2), frame_width, frame_height)


def yolo_hand_box_from_motion_roi(controller, frame, motion_roi_box):
    return yolo_hand_box_from_box(controller, frame, motion_roi_box)


def yolo_hand_box_from_box(controller, frame, search_box):
    if not ENABLE_YOLO or controller.yolo_model is None or search_box is None:
        return None

    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = clamp_roi_box(search_box, frame_width, frame_height)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    try:
        detections = controller.yolo_model.predict(crop, conf=YOLO_CONFIDENCE, verbose=False)
    except Exception as e:
        print(f"[WARN] YOLO hand detection failed: {e}", flush=True)
        return None

    best = None
    best_conf = -1.0
    names = getattr(controller.yolo_model, "names", {}) or {}

    for result in detections:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0]) if getattr(box, "cls", None) is not None else -1
            cls_name = str(names.get(cls_id, "")).lower()
            conf = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
            if names and cls_name not in {"hand", "hands"}:
                continue
            if conf > best_conf:
                best_conf = conf
                best = box.xyxy[0].tolist()

    if best is None:
        return None

    bx1, by1, bx2, by2 = best
    bx1 += x1
    bx2 += x1
    by1 += y1
    by2 += y1

    pad = max(bx2 - bx1, by2 - by1) * YOLO_ROI_PADDING_RATIO
    return clamp_roi_box((bx1 - pad, by1 - pad, bx2 + pad, by2 + pad), frame_width, frame_height)

def process_roi_with_landmark_remap(hands, frame, roi_box, roi_scale):
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = clamp_roi_box(roi_box, frame_width, frame_height)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    roi_height, roi_width = roi.shape[:2]
    upscaled = cv2.resize(
        roi,
        (roi_width * roi_scale, roi_height * roi_scale),
        interpolation=cv2.INTER_LINEAR,
    )
    roi_results = hands.process(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))
    if not has_hand(roi_results):
        return None

    for hand_landmarks in roi_results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            landmark.x = (x1 + landmark.x * roi_width) / frame_width
            landmark.y = (y1 + landmark.y * roi_height) / frame_height
    return roi_results


def hand_bbox_width_px(results, frame_width):
    if not has_hand(results):
        return 0
    xs = [lm.x for lm in results.multi_hand_landmarks[0].landmark]
    return int((max(xs) - min(xs)) * frame_width)

def hand_roi_box_from_results(results, frame_width, frame_height, padding_ratio):
    if not has_hand(results):
        return None

    landmarks = results.multi_hand_landmarks[0].landmark
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    x1 = min(xs) * frame_width
    y1 = min(ys) * frame_height
    x2 = max(xs) * frame_width
    y2 = max(ys) * frame_height

    box_w = x2 - x1
    box_h = y2 - y1
    size = max(box_w, box_h)

    if size <= 1:
        return None

    pad = size * padding_ratio

    return clamp_roi_box(
        (x1 - pad, y1 - pad, x2 + pad, y2 + pad),
        frame_width,
        frame_height,
    )


def print_status(payload):
    mode = "STANDBY" if payload["standby"] else "ACTIVE"
    print(
        f"[STATUS] {mode} bright={payload['brightness']} "
        f"MODE={payload['mode']} "
        f"gesture={payload['gesture'] or '-'} hand={payload['hand_detected']} "
        f"point_mode={payload['point_mode']} point={payload['point_status']} "
        f"pan={payload['pan_deg']:+.1f} tilt={payload['tilt_deg']:+.1f} "
        f"fps={payload['fps']}",
        flush=True,
    )
    if DEBUG_OUTPUT:
        emit_state(payload)


def draw_preview_overlay(frame, controller, fps, gesture, hand_detected, bbox, state):
    status = "STANDBY" if controller.standby else "ACTIVE"
    panel_h = 150
    panel = frame.copy()
    panel[:] = (18, 18, 18)
    lines = [
        f"{status} fps:{fps:.1f} hand:{hand_detected} bbox:{bbox}px proc:{state.get('processing_mode')}",
        f"gesture:{gesture or '-'} bright:{controller.brightness} last:{controller.last_brightness_gesture or '-'}",
        f"mode:{controller.mode} point_mode:{controller.point_mode} point:{controller.point_status} yolo:{controller.yolo_status}",
        f"wave:{state.get('wave_active')} span:{state.get('wave_motion_span', 0):.1f} turns:{state.get('wave_motion_turns', 0)} hits:{state.get('wave_open_palm_hits', 0)}/{state.get('wave_confirm_min_hits', 0)}",
        f"pan:{controller.preview_pan_deg:+.1f} tilt:{controller.preview_tilt_deg:+.1f} std:{controller.point_std_px:.0f}px",
    ]
    for i, text in enumerate(lines):
        cv2.putText(panel, text, (18, 30 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    show_point_marker = (
        controller.point_mode
        or controller.point_status in (
            "tracking",
            "tracking_depth",
            "tracking_depth_fallback",
            "tracking_2d_fallback",
            "locked",
        )
    )
    target = (controller.point_target or controller.point_display_target) if show_point_marker else None
    if target:
        tx, ty = int(target[0]), int(target[1])
        locked = controller.point_target is not None
        color = (0, 255, 80) if locked else (0, 210, 255)
        label = "LOCKED" if locked else f"{int(controller.point_stable_ratio * 100)}%"
        cv2.circle(frame, (tx, ty), 28, color, 2)
        cv2.circle(frame, (tx, ty), 6, color, -1)
        cv2.line(frame, (tx - 34, ty), (tx + 34, ty), color, 1)
        cv2.line(frame, (tx, ty - 34), (tx, ty + 34), color, 1)
        cv2.putText(frame, f"POINT {label}", (tx + 12, ty - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return cv2.vconcat([panel[:panel_h, :], frame])


def dispatch_hardware(led, controller, gesture, hw_state):
    # 서보 제어는 controller가 담당. dispatch는 LED + WAVE 시 서보 재기동만 처리.
    # POINT 잠금 전환 순간 1회만 LED 발화 (서보는 controller.update_point_target에서 이미 처리)
    if controller.point_status == "locked" and hw_state["point_status"] != "locked":
        led.spot_on(controller.brightness)

    if gesture == "WAVE":
        # 서보는 controller.wake()에서 이미 재기동 → LED만 켬
        if controller.mode == "Spot":
            led.spot_on(controller.brightness)
        else:
            led.mood_on(controller.brightness)
    elif gesture == "FIST":
        # controller.sleep()이 이미 servo.shutdown() 호출 (서보는 마지막 위치 유지)
        led.all_off()
    elif gesture in ("THUMBS_UP", "THUMBS_DOWN"):
        if controller.brightness != hw_state["brightness"]:
            if controller.mode == "Spot":
                led.spot_set(controller.brightness)
            else:
                led.mood_set(controller.brightness)
    elif gesture == "MODE_SWITCH":
        if controller.mode != hw_state["mode"]:
            if controller.mode == "Spot":
                led.mood_off()
                led.spot_on(controller.brightness)
            else:
                led.spot_off()
                led.mood_on(controller.brightness)

    hw_state["point_status"] = controller.point_status
    hw_state["mode"] = controller.mode
    hw_state["brightness"] = controller.brightness


def main():
    log_level = logging.DEBUG if DEBUG_OUTPUT else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    print(
        f"[PI] app.py algorithm runtime cam={CAM_INDEX} backend={CAMERA_BACKEND} "
        f"{FRAME_W}x{FRAME_H}@{TARGET_FPS} mirror={MIRROR}",
        flush=True,
    )
    print(
        f"[PI] output={'debug' if DEBUG_OUTPUT else 'normal'} "
        "(set PI_DEBUG=1 for STATE JSON and candidate logs)",
        flush=True,
    )

    controller = PiSmartLightController()
    led = LEDController()
    hw_state = {
        "point_status": controller.point_status,
        "mode": controller.mode,
        "brightness": controller.brightness,
    }
    controller.start_preloads()
    cap = open_camera()

    def _shutdown_hardware():
        try:
            cap.release()
        except Exception:
            pass
        if controller.servo is not None:
            try:
                controller.servo.shutdown()
            except Exception:
                pass
        try:
            led.shutdown()
        except Exception:
            pass
        if SHOW_PREVIEW:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    atexit.register(_shutdown_hardware)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    prev_gray = None
    centroid_history = []
    wave_candidate_frames = 0
    wave_motion_span = 0.0
    wave_motion_turns = 0
    wave_shape_hits = deque()
    no_motion_since = None
    hand_miss_frames = 0
    
    # New: tracked ROI / YOLO reacquire state
    last_hand_roi_box = None
    last_hand_roi_time = 0.0
    last_full_frame_time = 0.0
    last_yolo_time = 0.0
    last_inference_time = 0.0
    
    last_results = None
    last_results_time = 0.0
    last_hand_detected = False
    last_hand_bbox = 0
    last_gesture_state = controller.recognizer.extract_full_state(None)
    
    last_roi_fallback_time = 0.0
    skip_counter = 0
    fps_times = deque(maxlen=30)
    prev_time = time.time()
    last_status = 0.0
    last_state = {}

    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=MP_MODEL_COMPLEXITY,
        min_detection_confidence=MP_DET_CONF,
        min_tracking_confidence=MP_TRK_CONF,
        max_num_hands=1,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[PI] camera read failed", flush=True)
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            if MIRROR:
                frame = cv2.flip(frame, 1)
            clean_frame = frame.copy()

            now = time.time()
            fps_times.append(now - prev_time)
            prev_time = now
            fps = 1.0 / (sum(fps_times) / len(fps_times) + 1e-9)

            skip_processing = False
            if controller.standby:
                skip_counter += 1
                if skip_counter % 3 != 0:
                    skip_processing = True
            else:
                skip_counter = 0
                
            gesture = None
            results = last_results
            results_updated = False
            hand_detected = last_hand_detected
            hand_bbox = last_hand_bbox

            motion_detected = False
            motion_score = 0
            motion_roi_box = None
            processing_mode = "skipped_hold"
            gesture_zone_used = False
            yolo_roi_used = False
            motion_roi_used = False
            roi_box = None
            roi_scale = 1
            
            state = last_gesture_state.copy() if isinstance(last_gesture_state, dict) else {
                "gesture": None,
                "value": None,
            }
            state["processing_mode"] = "skipped_hold"
                
            # Limit expensive inference FPS separately from camera FPS.
            if not skip_processing:
                if controller.point_mode:
                    inference_fps_limit = POINT_INFERENCE_FPS
                elif controller.standby:
                    inference_fps_limit = STANDBY_INFERENCE_FPS
                else:
                    inference_fps_limit = ACTIVE_INFERENCE_FPS

                inference_interval = 1.0 / max(inference_fps_limit, 1.0)
                if now - last_inference_time < inference_interval:
                    skip_processing = True
                else:
                    last_inference_time = now

            if not skip_processing:
                small_frame = cv2.resize(frame, (160, 120))
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (9, 9), 0)

                if prev_gray is not None:
                    frame_delta = cv2.absdiff(prev_gray, gray)
                    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                    thresh = cv2.dilate(thresh, None, iterations=2)
                    motion_score = cv2.countNonZero(thresh)
                    min_motion_area = int((gray.shape[1] * gray.shape[0]) * 0.0015)

                    if motion_score > min_motion_area:
                        motion_detected = True
                        no_motion_since = None
                        motion_roi_box = motion_roi_box_from_thresh(thresh, frame.shape[1], frame.shape[0])

                        moments = cv2.moments(thresh)
                        if moments["m00"] > 0:
                            centroid_history.append(int(moments["m10"] / moments["m00"]))
                            if len(centroid_history) > 15:
                                centroid_history.pop(0)

                            if len(centroid_history) > 3:
                                span = max(centroid_history) - min(centroid_history)
                                min_span_threshold = max(4, int(gray.shape[1] * WAVE_MOTION_SPAN_RATIO))
                                x_steps = [
                                    centroid_history[i] - centroid_history[i - 1]
                                    for i in range(1, len(centroid_history))
                                ]
                                x_signs = [
                                    1 if step > 1 else -1
                                    for step in x_steps
                                    if abs(step) > 1
                                ]
                                x_turns = sum(
                                    1 for i in range(1, len(x_signs)) if x_signs[i] != x_signs[i - 1]
                                )

                                wave_motion_span = float(span)
                                wave_motion_turns = x_turns

                                if controller.standby and span > min_span_threshold and x_turns >= 2:
                                    if DEBUG_OUTPUT:
                                        print(
                                            f"[DEBUG] WAVE CANDIDATE triggered "
                                            f"(span: {span:.1f} > threshold: {min_span_threshold})",
                                            flush=True,
                                        )
                                    wave_candidate_frames = WAVE_CANDIDATE_FRAMES
                                    centroid_history.clear()

                if not motion_detected and no_motion_since is None:
                    no_motion_since = time.time()

                prev_gray = gray

                active_deadline_alive = not controller.standby and time.time() < controller.active_until
                skip_inference = (
                    not motion_detected
                    and wave_candidate_frames <= 0
                    and not active_deadline_alive
                )

                if not skip_inference:
                    now_infer = time.time()
                    frame_height, frame_width = frame.shape[:2]
                    new_results = None

                    # 1) If we recently saw a hand, try tracked hand ROI first.
                    tracked_roi_valid = (
                        not controller.standby
                        and not controller.point_mode
                        and last_hand_roi_box is not None
                        and (now_infer - last_hand_roi_time) <= TRACK_ROI_TTL
                        and (now_infer - last_full_frame_time) < FULL_FRAME_REACQUIRE_INTERVAL
                    )

                    if tracked_roi_valid:
                        roi_results = process_roi_with_landmark_remap(
                            hands,
                            frame,
                            last_hand_roi_box,
                            TRACK_ROI_SCALE,
                        )
                        if has_hand(roi_results):
                            new_results = roi_results
                            processing_mode = "tracked_hand_roi"
                            roi_box = last_hand_roi_box
                            roi_scale = TRACK_ROI_SCALE

                    # 2) If tracked ROI failed, try motion ROI with MediaPipe.
                    if (
                        new_results is None
                        and ENABLE_MOTION_ROI
                        and motion_roi_box is not None
                        and not controller.standby
                        and not controller.point_mode
                    ):
                        roi_results = process_roi_with_landmark_remap(
                            hands,
                            frame,
                            motion_roi_box,
                            ROI_SCALE,
                        )
                        motion_roi_used = True
                        roi_box = motion_roi_box
                        roi_scale = ROI_SCALE

                        if has_hand(roi_results):
                            new_results = roi_results
                            processing_mode = "motion_roi"

                    # 3) If still no hand, use YOLO as a reacquire detector.
                    should_try_yolo = (
                        new_results is None
                        and not controller.point_mode
                        and ENABLE_YOLO
                        and controller.yolo_model is not None
                        and hand_miss_frames >= YOLO_AFTER_MISSES
                        and (now_infer - last_yolo_time) >= YOLO_REACQUIRE_INTERVAL
                    )

                    if should_try_yolo:
                        last_yolo_time = now_infer

                        if motion_roi_box is not None:
                            yolo_search_box = motion_roi_box
                        elif ENABLE_GESTURE_ZONE_ROI:
                            yolo_search_box = gesture_zone_box(frame_width, frame_height)
                        else:
                            yolo_search_box = (0, 0, frame_width, frame_height)

                        yolo_roi_box = yolo_hand_box_from_box(controller, frame, yolo_search_box)

                        if yolo_roi_box is not None:
                            yolo_roi_used = True
                            roi_box = yolo_roi_box
                            roi_scale = YOLO_ROI_SCALE

                            roi_results = process_roi_with_landmark_remap(
                                hands,
                                frame,
                                yolo_roi_box,
                                YOLO_ROI_SCALE,
                            )
                            if has_hand(roi_results):
                                new_results = roi_results
                                processing_mode = "yolo_reacquire_roi"

                    # 4) Full-frame reacquire periodically or when all ROI paths fail.
                    if new_results is None:
                        processing_mode = "full_frame"
                        new_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        last_full_frame_time = now_infer

                    results = new_results
                    results_updated = True

                # Update hand state only when we actually ran inference.
                if results_updated:
                    hand_detected = has_hand(results)

                    if hand_detected:
                        hand_miss_frames = 0

                        updated_roi = hand_roi_box_from_results(
                            results,
                            frame.shape[1],
                            frame.shape[0],
                            TRACK_ROI_PADDING_RATIO,
                        )
                        if updated_roi is not None:
                            last_hand_roi_box = updated_roi
                            last_hand_roi_time = time.time()
                    else:
                        hand_miss_frames += 1
                        results = None
                        hand_detected = False
                else:
                    # No new inference this frame. Keep previous hand state.
                    hand_detected = last_hand_detected
                    hand_bbox = last_hand_bbox
                    results = last_results

                if results_updated:
                    hand_bbox = hand_bbox_width_px(results, frame.shape[1])
                else:
                    hand_bbox = last_hand_bbox

                # Save cache for skipped frames.
                last_results = results
                last_hand_detected = hand_detected
                last_hand_bbox = hand_bbox
                if results is not None and has_hand(results):
                    last_results_time = time.time()

                controller.update_activity_timeout(hand_detected)

                # if hand_detected and SHOW_PREVIEW and results:
                #     mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

                if results_updated:
                    state = (
                        controller.recognizer.extract_full_state(results)
                        if results
                        else controller.recognizer.extract_full_state(None)
                    )
                    last_gesture_state = state.copy()
                else:
                    state = (
                        last_gesture_state.copy()
                        if isinstance(last_gesture_state, dict)
                        else controller.recognizer.extract_full_state(None)
                    )

                    # Do not replay most gestures on skipped frames,
                    # because brightness/mode commands could repeat.
                    # Exception: keep POINT during point mode so target stability does not reset.
                    state["gesture"] = None
                    state["value"] = None
                    state["confirmed_gesture"] = None

                    state["processing_mode"] = "skipped_hold"
                tick_gesture_allowed = (
                    not ENABLE_HAND_SIZE_GATING
                    or hand_bbox >= MIN_FINE_GESTURE_HAND_BBOX_WIDTH_PX
                )
                fine_gesture_allowed = tick_gesture_allowed
                is_shaking = False
                if controller.standby and wave_candidate_frames > 0:
                    wave_candidate_frames -= 1
                    is_shaking = True
                elif not controller.standby:
                    wave_candidate_frames = 0

                # WAVE is only used to wake from STANDBY.
                # Once ACTIVE, ignore WAVE completely so it does not interfere with other gestures.
                if controller.standby:
                    if is_shaking and state.get("gesture") == "WAVE":
                        wave_shape_hits.append(time.time())

                    while wave_shape_hits and time.time() - wave_shape_hits[0] > WAVE_CONFIRM_WINDOW_SECONDS:
                        wave_shape_hits.popleft()

                    wave_confirmed = (
                        is_shaking
                        and hand_detected
                        and len(wave_shape_hits) >= WAVE_CONFIRM_MIN_HITS
                    )

                    if wave_confirmed:
                        state["gesture"] = "WAVE"
                        state["value"] = None
                    elif state.get("gesture") == "WAVE":
                        state["gesture"] = None
                else:
                    wave_shape_hits.clear()
                    if state.get("gesture") == "WAVE":
                        state["gesture"] = None
                        state["value"] = None
                        state["confirmed_gesture"] = None

                current_gesture = state.get("gesture")
                if controller.standby and current_gesture in ("THUMBS_UP", "THUMBS_DOWN"):
                    if DEBUG_OUTPUT:
                        print("[THUMBS BLOCKED] standby mode, suppress brightness gesture", flush=True)
                    state["gesture"] = None
                    state["value"] = None
                    current_gesture = None

                if not controller.standby and current_gesture == "POINT" and not controller.point_mode:
                    if DEBUG_OUTPUT:
                        print("[POINT BLOCKED] POINT ignored because POINT_MODE is not enabled", flush=True)
                    state["gesture"] = None
                    state["value"] = None
                    state["confirmed_gesture"] = None
                    current_gesture = None

                gesture = controller.apply_gesture(current_gesture, results, clean_frame, is_shaking)
                dispatch_hardware(led, controller, gesture, hw_state)

                state.update(
                    {
                        "python_standby": controller.standby,
                        "point_mode_enabled": controller.point_mode,
                        "keep_awake_for_testing": controller.state_payload(fps, gesture, hand_detected, hand_bbox, is_shaking)["keep_awake"],
                        "active_hold_remaining": max(0.0, controller.active_until - time.time()) if not controller.standby else 0.0,
                        "motion_score": motion_score,
                        "wave_motion_active": is_shaking,
                        "wave_motion_span": wave_motion_span,
                        "wave_motion_turns": wave_motion_turns,
                        "wave_open_palm_hits": len(wave_shape_hits),
                        "wave_confirm_min_hits": WAVE_CONFIRM_MIN_HITS,
                        "processing_mode": processing_mode,
                        "hand_detected": hand_detected,
                        "gesture_zone_used": gesture_zone_used,
                        "yolo_roi_used": yolo_roi_used,
                        "yolo_hand_status": controller.yolo_status,
                        "motion_roi_used": motion_roi_used,
                        "roi_box": roi_box,
                        "roi_scale": roi_scale,
                        "hand_bbox_width_px": hand_bbox,
                        "fine_gesture_allowed": fine_gesture_allowed,
                        "tick_gesture_allowed": tick_gesture_allowed,
                    }
                )

                last_state = state
            else:
                controller.update_activity_timeout(False)
                last_state = state
                
            # Draw cached landmarks on every preview frame to reduce flicker.
            if SHOW_PREVIEW:
                draw_results = results if results is not None else last_results
                draw_hand_detected = hand_detected or last_hand_detected

                if draw_hand_detected and draw_results and has_hand(draw_results):
                    mp_draw.draw_landmarks(
                        frame,
                        draw_results.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS,
                    )
            if now - last_status >= STATUS_INTERVAL:
                last_status = now
                payload = controller.state_payload(fps, gesture, hand_detected, hand_bbox, last_state.get("wave_motion_active", False))
                payload.update(
                    {
                        "processing_mode": last_state.get("processing_mode"),
                        "motion_score": last_state.get("motion_score", 0),
                        "wave_motion_span": last_state.get("wave_motion_span", 0.0),
                        "wave_motion_turns": last_state.get("wave_motion_turns", 0),
                        "wave_open_palm_hits": last_state.get("wave_open_palm_hits", 0),
                        "wave_confirm_min_hits": last_state.get("wave_confirm_min_hits", WAVE_CONFIRM_MIN_HITS),
                        "gesture_zone_used": last_state.get("gesture_zone_used", False),
                        "yolo_roi_used": last_state.get("yolo_roi_used", False),
                        "motion_roi_used": last_state.get("motion_roi_used", False),
                        "roi_box": last_state.get("roi_box"),
                        "roi_scale": last_state.get("roi_scale", 1),
                    }
                )
                print_status(payload)

            if SHOW_PREVIEW:
                preview = draw_preview_overlay(frame, controller, fps, gesture, hand_detected, hand_bbox, last_state)
                cv2.imshow("Pi Smart Light Runtime", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    # 정상 종료 및 예외 모두 atexit 핸들러(_shutdown_hardware)가 처리


if __name__ == "__main__":
    main()



