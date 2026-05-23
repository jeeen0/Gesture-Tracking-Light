import os
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return str(cwd_path.resolve())

    return str((PROJECT_ROOT / path).resolve())


_CAM_INDEX_RAW = os.environ.get("PI_CAM_INDEX", "0")
try:
    CAM_INDEX = int(_CAM_INDEX_RAW)
except ValueError:
    CAM_INDEX = _CAM_INDEX_RAW

CAMERA_BACKEND = os.environ.get("PI_CAMERA_BACKEND", "rpicam-vid").lower()
FRAME_W = int(os.environ.get("PI_FRAME_W", "640"))
FRAME_H = int(os.environ.get("PI_FRAME_H", "360"))
TARGET_FPS = int(os.environ.get("PI_TARGET_FPS", "20"))
MIRROR = os.environ.get("PI_MIRROR", "1") != "0"
SHOW_PREVIEW = os.environ.get("PI_SHOW_PREVIEW", "1") == "1"
PREVIEW_SCALE = max(0.1, float(os.environ.get("PI_PREVIEW_SCALE", "1.0")))
SHOW_DEPTH = os.environ.get("PI_SHOW_DEPTH", "0") == "1"
DEBUG_OUTPUT = os.environ.get("PI_DEBUG", "0") == "1"

# Video recording. Set PI_SAVE_VIDEO=1 to save the runtime stream.
SAVE_VIDEO = os.environ.get("PI_SAVE_VIDEO", "0") == "1"
_DEFAULT_SAVE_VIDEO_NAME = f"recordings/pi_runtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
SAVE_VIDEO_PATH = resolve_project_path(os.environ.get("PI_SAVE_VIDEO_PATH", _DEFAULT_SAVE_VIDEO_NAME))
SAVE_VIDEO_FOURCC = os.environ.get("PI_SAVE_VIDEO_FOURCC", "mp4v")
SAVE_VIDEO_FPS = float(os.environ.get("PI_SAVE_VIDEO_FPS", str(TARGET_FPS)))
SAVE_VIDEO_OVERLAY = os.environ.get("PI_SAVE_VIDEO_OVERLAY", "1") != "0"

MP_DET_CONF = float(os.environ.get("PI_MP_DET_CONF", "0.30"))
MP_TRK_CONF = float(os.environ.get("PI_MP_TRK_CONF", "0.30"))

# MediaPipe Hands model.
# 0 = lite / faster, 1 = full / default-ish, 2 = heavier if supported.
MP_MODEL_COMPLEXITY = int(os.environ.get("PI_MP_MODEL_COMPLEXITY", "0"))

# Limit expensive AI inference separately from camera FPS.
# Camera can still run at TARGET_FPS, but MediaPipe/ROI/YOLO processing is throttled.
DEFAULT_INFERENCE_FPS = float(os.environ.get("PI_INFERENCE_FPS", "20"))
STANDBY_INFERENCE_FPS = float(os.environ.get("PI_STANDBY_INFERENCE_FPS", "5"))
ACTIVE_INFERENCE_FPS = float(os.environ.get("PI_ACTIVE_INFERENCE_FPS", str(DEFAULT_INFERENCE_FPS)))
POINT_INFERENCE_FPS = float(os.environ.get("PI_POINT_INFERENCE_FPS", "20"))

ACTIVE_TIMEOUT_SECONDS = float(os.environ.get("PI_ACTIVE_TIMEOUT", "20.0"))
KEEP_AWAKE = os.environ.get("PI_KEEP_AWAKE", "1") != "0"

ENABLE_POINTING = os.environ.get("PI_ENABLE_POINT", "1") != "0"
PRELOAD_POINTING = os.environ.get("PI_PRELOAD_POINT", "1") != "0"

# 포인팅 모드에서도 ROI 경로 사용 여부.
# - POINT_USE_ROI: tracked/motion ROI (padding 큼, 안전). 기본 ON.
# - POINT_USE_YOLO_ROI: YOLO ROI (padding 작음, 손가락 끝이 잘릴 위험). 기본 OFF.
ENABLE_POINT_ROI = os.environ.get("PI_POINT_USE_ROI", "1") != "0"
ENABLE_POINT_YOLO_ROI = os.environ.get("PI_POINT_USE_YOLO_ROI", "0") != "0"

# YOLO ROI
ENABLE_YOLO = os.environ.get("PI_ENABLE_YOLO", "0") != "0"
PRELOAD_YOLO = os.environ.get("PI_PRELOAD_YOLO", "0") != "0"
YOLO_MODEL_PATH = resolve_project_path(os.environ.get("PI_YOLO_MODEL", "models/hand_yolo.pt"))
YOLO_CONFIDENCE = float(os.environ.get("PI_YOLO_CONF", "0.25"))
YOLO_ROI_SCALE = int(os.environ.get("PI_YOLO_ROI_SCALE", "2"))
YOLO_ROI_PADDING_RATIO = float(os.environ.get("PI_YOLO_ROI_PADDING", "0.25"))

# 기본 ROI (Tracked hand ROI): once a hand is found, use this smaller ROI first.
TRACK_ROI_SCALE = int(os.environ.get("PI_TRACK_ROI_SCALE", "2"))
TRACK_ROI_PADDING_RATIO = float(os.environ.get("PI_TRACK_ROI_PADDING", "0.65"))
TRACK_ROI_TTL = float(os.environ.get("PI_TRACK_ROI_TTL", "1.0"))
FULL_FRAME_REACQUIRE_INTERVAL = float(os.environ.get("PI_FULL_FRAME_REACQUIRE_INTERVAL", "0.50"))

# YOLO should be used as a reacquire detector, not every frame.
YOLO_REACQUIRE_INTERVAL = float(os.environ.get("PI_YOLO_REACQUIRE_INTERVAL", "0.70"))
YOLO_AFTER_MISSES = int(os.environ.get("PI_YOLO_AFTER_MISSES", "2"))

ENABLE_GESTURE_ZONE_ROI = os.environ.get("PI_ENABLE_GESTURE_ZONE_ROI", "1") != "0"
ENABLE_MOTION_ROI = os.environ.get("PI_ENABLE_MOTION_ROI", "1") != "0"
GESTURE_ZONE = {
    "x_min_ratio": float(os.environ.get("PI_GESTURE_ZONE_X_MIN", "0.10")),
    "y_min_ratio": float(os.environ.get("PI_GESTURE_ZONE_Y_MIN", "0.05")),
    "x_max_ratio": float(os.environ.get("PI_GESTURE_ZONE_X_MAX", "0.90")),
    "y_max_ratio": float(os.environ.get("PI_GESTURE_ZONE_Y_MAX", "0.95")),
}
ROI_SCALE = int(os.environ.get("PI_ROI_SCALE", "2"))
ROI_FALLBACK_AFTER_MISSES = int(os.environ.get("PI_ROI_FALLBACK_AFTER_MISSES", "1"))
ROI_FALLBACK_INTERVAL = float(os.environ.get("PI_ROI_FALLBACK_INTERVAL", "0.08"))
MOTION_ROI_PADDING_RATIO = float(os.environ.get("PI_MOTION_ROI_PADDING", "0.35"))
MIN_MOTION_ROI_SIZE = int(os.environ.get("PI_MIN_MOTION_ROI_SIZE", "48"))

ENABLE_HAND_SIZE_GATING = os.environ.get("PI_ENABLE_HAND_SIZE_GATING", "1") != "0"
MIN_FINE_GESTURE_HAND_BBOX_WIDTH_PX = int(os.environ.get("PI_MIN_FINE_GESTURE_BBOX", "35"))

BRIGHTNESS_STEP = int(os.environ.get("PI_BRIGHTNESS_STEP", "2"))
BRIGHTNESS_UPDATE_INTERVAL = float(os.environ.get("PI_BRIGHTNESS_INTERVAL", "0.12"))
COMMAND_HOLD_SECONDS = float(os.environ.get("PI_COMMAND_HOLD", "0.5"))
MIN_BRIGHTNESS = 0
MAX_BRIGHTNESS = 100

STATUS_INTERVAL = float(os.environ.get("PI_STATUS_INTERVAL", "1.0"))


def _resolve_state_file(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


STATE_FILE = _resolve_state_file(os.environ.get("PI_STATE_FILE", "pi_state.json"))
WAVE_CANDIDATE_FRAMES = int(os.environ.get("PI_WAVE_CANDIDATE_FRAMES", "8")) # 기존 22
WAVE_MOTION_SPAN_RATIO = float(os.environ.get("PI_WAVE_MOTION_SPAN_RATIO", "0.03"))
WAVE_CONFIRM_WINDOW_SECONDS = float(os.environ.get("PI_WAVE_CONFIRM_WINDOW", "0.8"))
WAVE_CONFIRM_MIN_HITS = int(os.environ.get("PI_WAVE_CONFIRM_HITS", "2"))

# Fisheye 카메라 보정 (.npz). 파일이 없으면 보정 비활성, 기본 fx 추정값 사용.
FISHEYE_CALIB_PATH = resolve_project_path(
    os.environ.get("PI_FISHEYE_CALIB", "src/calibration/fisheye_undistort_map.npz")
)
ENABLE_FISHEYE_UNDISTORT = os.environ.get("PI_ENABLE_FISHEYE", "1") != "0"

# 카메라-서보 방향 부호 보정. 짐벌 조립 방향에 따라 +1 또는 -1.
# 카메라 오른쪽(+pan_deg)이 서보 +방향과 일치하면 +1, 반대면 -1.
# 카메라 아래쪽(+tilt_deg)이 서보 +방향과 일치하면 +1, 반대면 -1.
SERVO_PAN_SIGN = int(os.environ.get("PI_SERVO_PAN_SIGN", "1"))
SERVO_TILT_SIGN = int(os.environ.get("PI_SERVO_TILT_SIGN", "-1"))

# Pointing calibration. CENTER is the servo angle that points light at camera center.
SERVO_PAN_CENTER = float(os.environ.get("PI_SERVO_PAN_CENTER", "90.0"))
SERVO_TILT_CENTER = float(os.environ.get("PI_SERVO_TILT_CENTER", "135.0"))

POINT_PAN_GAIN = float(os.environ.get("PI_POINT_PAN_GAIN", "1.0"))
POINT_TILT_GAIN = float(os.environ.get("PI_POINT_TILT_GAIN", "1.0"))

POINT_PAN_OFFSET_DEG = float(os.environ.get("PI_POINT_PAN_OFFSET_DEG", "0.0"))
POINT_TILT_OFFSET_DEG = float(os.environ.get("PI_POINT_TILT_OFFSET_DEG", "0.0"))
