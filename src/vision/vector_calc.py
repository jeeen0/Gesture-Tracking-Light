"""
벡터 계산 - MediaPipe 손 랜드마크에서 포인팅 방향 벡터 추출
+ 카메라 좌표계 → 짐벌(Pan/Tilt) 각도 변환

중간보고서의 Solution(A) 비율 기반 추정 방식 구현:
    Z = (실제 손 길이 × 카메라 초점거리) / 픽셀 상 손 길이
"""
import logging
import math
from typing import Optional

import numpy as np

from src.config import (
    HAND_REAL_LENGTH_M,
    PAN_MIN_DEG, PAN_MAX_DEG,
    TILT_MIN_DEG, TILT_MAX_DEG,
)

log = logging.getLogger(__name__)


# 카메라 내부 파라미터 (캘리브레이션으로 정확한 값 구해야 함)
# Pi Camera Module 3 Wide 기준 임시값
CAMERA_FOCAL_PX = 500.0       # 픽셀 단위 초점거리
CAMERA_CX = 320.0             # 광학 중심 x (640 해상도 기준)
CAMERA_CY = 240.0             # 광학 중심 y

# MediaPipe 랜드마크 인덱스
LM_WRIST = 0
LM_INDEX_MCP = 5
LM_INDEX_TIP = 8


def estimate_depth(hand_landmarks_px, hand_real_length_m: float = HAND_REAL_LENGTH_M) -> float:
    """카메라-손 사이 거리 Z (미터) 추정.
    
    삼각 유사성:
        실제 손 길이 / 실제 거리 = 픽셀 손 길이 / 초점거리
        → Z = (실제 길이 × 초점거리) / 픽셀 길이
    
    hand_landmarks_px: 픽셀 좌표 (N, 3) numpy 배열. [x, y, z_rel]
    """
    wrist = hand_landmarks_px[LM_WRIST][:2]
    index_mcp = hand_landmarks_px[LM_INDEX_MCP][:2]
    pixel_length = np.linalg.norm(index_mcp - wrist)
    
    if pixel_length < 1:
        return 1.0  # fallback
    
    z = (hand_real_length_m * CAMERA_FOCAL_PX) / pixel_length
    return float(z)


def calc_pointing_vector(hand_landmarks_px, z_estimated: float) -> np.ndarray:
    """검지 끝 - 손목 벡터를 카메라 좌표계의 3D 방향으로 변환.
    
    Returns: (vx, vy, vz) 정규화된 3D 방향 벡터 (카메라 좌표계)
    """
    # 픽셀 좌표 → 카메라 좌표 변환
    def px_to_cam(px, py, z):
        x_cam = (px - CAMERA_CX) * z / CAMERA_FOCAL_PX
        y_cam = (py - CAMERA_CY) * z / CAMERA_FOCAL_PX
        z_cam = z
        return np.array([x_cam, y_cam, z_cam])
    
    wrist_3d = px_to_cam(hand_landmarks_px[LM_WRIST][0],
                         hand_landmarks_px[LM_WRIST][1],
                         z_estimated)
    tip_3d = px_to_cam(hand_landmarks_px[LM_INDEX_TIP][0],
                       hand_landmarks_px[LM_INDEX_TIP][1],
                       z_estimated)
    
    direction = tip_3d - wrist_3d
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return direction / norm


def vector_to_pan_tilt(direction: np.ndarray) -> tuple[float, float]:
    """3D 방향 벡터 → 짐벌 Pan/Tilt 각도 변환.
    
    Pan(yaw)   = atan2(x, z)   - 좌우 회전 (수평)
    Tilt(pitch) = atan2(-y, sqrt(x²+z²))  - 상하 회전 (수직)
    
    카메라 좌표계: +x=오른쪽, +y=아래, +z=앞.
    짐벌 0° = 정면, Pan +방향 = 오른쪽, Tilt +방향 = 위.
    
    Returns: (pan_deg, tilt_deg) 짐벌 안전 범위로 클램핑됨.
    """
    x, y, z = direction
    
    # Pan: 90° = 정면. 오른쪽 보면 90+α, 왼쪽 보면 90-α
    pan_rad = math.atan2(x, z) if z > 0.01 else 0.0
    pan_deg = 90.0 + math.degrees(pan_rad)
    
    # Tilt: 90° = 수평. 위 보면 90-α, 아래 보면 90+α 
    horizontal = math.sqrt(x * x + z * z)
    tilt_rad = math.atan2(-y, horizontal) if horizontal > 0.01 else 0.0
    tilt_deg = 90.0 - math.degrees(tilt_rad)
    
    # 안전 범위로 클램핑
    pan_deg = max(PAN_MIN_DEG, min(PAN_MAX_DEG, pan_deg))
    tilt_deg = max(TILT_MIN_DEG, min(TILT_MAX_DEG, tilt_deg))
    
    return pan_deg, tilt_deg


def hand_landmarks_to_pixels(landmarks, img_width: int, img_height: int) -> np.ndarray:
    """MediaPipe 정규화 좌표 (0~1) → 픽셀 좌표 (N, 3) 변환.
    
    landmarks: mediapipe HandLandmarkerResult.hand_landmarks[i] 형태
               각 랜드마크는 .x, .y, .z 속성 가짐.
    """
    pts = np.zeros((21, 3))
    for i, lm in enumerate(landmarks):
        pts[i, 0] = lm.x * img_width
        pts[i, 1] = lm.y * img_height
        pts[i, 2] = lm.z  # 상대 깊이 (정규화됨)
    return pts


def hand_pixel_size(landmarks_px: np.ndarray) -> float:
    """손목 - 검지 MCP 픽셀 거리 (손 크기 추정용)."""
    wrist = landmarks_px[LM_WRIST][:2]
    mcp = landmarks_px[LM_INDEX_MCP][:2]
    return float(np.linalg.norm(mcp - wrist))