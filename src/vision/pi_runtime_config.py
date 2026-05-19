import os
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
FRAME_H = int(os.environ.get("PI_FRAME_H", "480"))
TARGET_FPS = int(os.environ.get("PI_TARGET_FPS", "30"))
MIRROR = os.environ.get("PI_MIRROR", "1") != "0"
SHOW_PREVIEW = os.environ.get("PI_SHOW_PREVIEW", "0") == "1"
DEBUG_OUTPUT = os.environ.get("PI_DEBUG", "0") == "1"

MP_DET_CONF = float(os.environ.get("PI_MP_DET_CONF", "0.30"))
MP_TRK_CONF = float(os.environ.get("PI_MP_TRK_CONF", "0.30"))

ACTIVE_TIMEOUT_SECONDS = float(os.environ.get("PI_ACTIVE_TIMEOUT", "20.0"))
KEEP_AWAKE = os.environ.get("PI_KEEP_AWAKE", "1") != "0"

ENABLE_POINTING = os.environ.get("PI_ENABLE_POINT", "1") != "0"
PRELOAD_POINTING = os.environ.get("PI_PRELOAD_POINT", "1") != "0"

ENABLE_YOLO = os.environ.get("PI_ENABLE_YOLO", "1") != "0"
PRELOAD_YOLO = os.environ.get("PI_PRELOAD_YOLO", "1") != "0"
YOLO_MODEL_PATH = resolve_project_path(os.environ.get("PI_YOLO_MODEL", "models/hand_yolo.pt"))
YOLO_CONFIDENCE = float(os.environ.get("PI_YOLO_CONF", "0.25"))
YOLO_ROI_SCALE = int(os.environ.get("PI_YOLO_ROI_SCALE", "2"))
YOLO_ROI_PADDING_RATIO = float(os.environ.get("PI_YOLO_ROI_PADDING", "0.25"))

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
STATE_FILE = os.environ.get("PI_STATE_FILE", "pi_state.json")
WAVE_CANDIDATE_FRAMES = int(os.environ.get("PI_WAVE_CANDIDATE_FRAMES", "22"))
WAVE_MOTION_SPAN_RATIO = float(os.environ.get("PI_WAVE_MOTION_SPAN_RATIO", "0.03"))
WAVE_CONFIRM_WINDOW_SECONDS = float(os.environ.get("PI_WAVE_CONFIRM_WINDOW", "0.8"))
WAVE_CONFIRM_MIN_HITS = int(os.environ.get("PI_WAVE_CONFIRM_HITS", "2"))

