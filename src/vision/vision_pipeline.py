"""
비전 파이프라인 - MediaPipe Hands 기반 손 랜드마크 추출 + 제스처 분류

지원 제스처 (중간보고서 9페이지 기준):
- WAVE: 손바닥 좌우 흔들기 (대기 ↔ 활성 전환)
- POINT: 검지만 핀 상태 (조명 방향 제어)
- PINCH: 엄지 + 검지 (밝기 조절)
- SHAKA: 엄지 + 새끼손가락 (모드 전환 spot ↔ mood)
- FIST: 주먹 (조명 OFF)
"""
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import mediapipe as mp
    import cv2
except ImportError:
    mp = None
    cv2 = None

from src.config import (
    WAVE_FRAMES, WAVE_MIN_AMPLITUDE,
    PINCH_ENTRY_RATIO, PINCH_EXIT_RATIO,
    PINCH_BRIGHT_MIN_RATIO, PINCH_BRIGHT_MAX_RATIO,
)
from src.vision.vector_calc import (
    hand_landmarks_to_pixels, hand_pixel_size,
    estimate_depth, calc_pointing_vector, vector_to_pan_tilt,
    LM_WRIST, LM_INDEX_MCP, LM_INDEX_TIP,
)

log = logging.getLogger(__name__)


# MediaPipe 손 랜드마크 인덱스
LM_THUMB_TIP = 4
LM_MIDDLE_TIP = 12
LM_RING_TIP = 16
LM_PINKY_TIP = 20
FINGER_TIPS = [LM_INDEX_TIP, LM_MIDDLE_TIP, LM_RING_TIP, LM_PINKY_TIP]
FINGER_MCPS = [LM_INDEX_MCP, 9, 13, 17]


@dataclass
class GestureResult:
    """비전 파이프라인 결과."""
    gesture: str                       # "WAVE" | "POINT" | "PINCH" | "SHAKA" | "FIST" | "NONE"
    pan_deg: Optional[float] = None    # POINT 시 짐벌 목표 Pan
    tilt_deg: Optional[float] = None   # POINT 시 짐벌 목표 Tilt
    brightness: Optional[float] = None # PINCH 시 0~100 밝기
    confidence: float = 0.0


def is_finger_extended(landmarks_px: np.ndarray, tip_idx: int, mcp_idx: int) -> bool:
    """손가락이 펴졌는지 (tip이 mcp보다 손목에서 멀면 펴진 것)."""
    wrist = landmarks_px[LM_WRIST][:2]
    tip_dist = np.linalg.norm(landmarks_px[tip_idx][:2] - wrist)
    mcp_dist = np.linalg.norm(landmarks_px[mcp_idx][:2] - wrist)
    return tip_dist > mcp_dist * 1.15  # 15% 이상 멀어야 펴진 걸로


def is_thumb_extended(landmarks_px: np.ndarray) -> bool:
    """엄지는 다른 손가락과 방향이 달라서 별도 처리."""
    wrist = landmarks_px[LM_WRIST][:2]
    thumb_tip = landmarks_px[LM_THUMB_TIP][:2]
    thumb_mcp = landmarks_px[2][:2]  # THUMB_MCP
    return np.linalg.norm(thumb_tip - wrist) > np.linalg.norm(thumb_mcp - wrist) * 1.1


class VisionPipeline:
    """카메라 입력 → MediaPipe → 제스처 분류."""
    
    def __init__(self):
        if mp is None:
            log.warning("mediapipe not available - vision in DRY mode")
            self.hands = None
        else:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
            log.info("MediaPipe Hands initialized")
        
        # Wave 감지용 히스토리 (손목 x좌표)
        self._wrist_x_history = deque(maxlen=WAVE_FRAMES)
        # Pinch 상태 (히스테리시스)
        self._in_pinch = False
        # 손 정지 감지용 (위치 변화량)
        self._prev_wrist_pos: Optional[np.ndarray] = None
        self._still_threshold = 15.0  # 픽셀
    
    def _classify_gesture(self, landmarks_px: np.ndarray) -> str:
        """랜드마크 → 제스처 분류 (규칙 기반)."""
        # 손가락 펴짐 상태 (검지, 중지, 약지, 새끼)
        fingers = [is_finger_extended(landmarks_px, tip, mcp)
                   for tip, mcp in zip(FINGER_TIPS, FINGER_MCPS)]
        thumb = is_thumb_extended(landmarks_px)
        
        # FIST: 모두 접힘
        if not any(fingers) and not thumb:
            return "FIST"
        
        # SHAKA: 엄지 + 새끼만 펴짐
        if thumb and not fingers[0] and not fingers[1] and not fingers[2] and fingers[3]:
            return "SHAKA"
        
        # POINT: 검지만 펴짐 (다른 손가락 접힘)
        if fingers[0] and not fingers[1] and not fingers[2] and not fingers[3]:
            return "POINT"
        
        # PINCH 판정 (엄지-검지 거리)
        hand_size = hand_pixel_size(landmarks_px)
        if hand_size > 5:
            thumb_tip = landmarks_px[LM_THUMB_TIP][:2]
            index_tip = landmarks_px[LM_INDEX_TIP][:2]
            pinch_dist = np.linalg.norm(thumb_tip - index_tip)
            ratio = pinch_dist / hand_size
            # 히스테리시스
            if not self._in_pinch and ratio < PINCH_ENTRY_RATIO:
                self._in_pinch = True
            elif self._in_pinch and ratio > PINCH_EXIT_RATIO:
                self._in_pinch = False
            if self._in_pinch:
                return "PINCH"
        
        # 모두 펴짐: 손바닥 (Wave 후보)
        if all(fingers):
            return "PALM"
        
        return "NONE"
    
    def _check_wave(self, wrist_x_px: float) -> bool:
        """좌우 흔들기 패턴 검출."""
        self._wrist_x_history.append(wrist_x_px)
        if len(self._wrist_x_history) < WAVE_FRAMES:
            return False
        
        xs = np.array(self._wrist_x_history)
        amplitude = xs.max() - xs.min()
        if amplitude < WAVE_MIN_AMPLITUDE:
            return False
        
        # 방향 변화 횟수 (zero-crossing)
        diffs = np.diff(xs)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        return sign_changes >= 3
    
    def _check_hand_still(self, wrist_pos: np.ndarray) -> bool:
        """손목 위치가 거의 정지인지."""
        if self._prev_wrist_pos is None:
            self._prev_wrist_pos = wrist_pos
            return False
        moved = np.linalg.norm(wrist_pos - self._prev_wrist_pos)
        self._prev_wrist_pos = wrist_pos
        return moved < self._still_threshold
    
    def process_frame(self, frame_bgr) -> GestureResult:
        """카메라 프레임 한 장 처리 → 제스처 + 부가 정보 반환.
        
        frame_bgr: OpenCV BGR 이미지 (H, W, 3)
        """
        if self.hands is None or cv2 is None or frame_bgr is None:
            return GestureResult(gesture="NONE")
        
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        
        if not result.multi_hand_landmarks:
            self._wrist_x_history.clear()
            self._prev_wrist_pos = None
            return GestureResult(gesture="NONE")
        
        landmarks = result.multi_hand_landmarks[0].landmark
        pts_px = hand_landmarks_to_pixels(landmarks, w, h)
        
        # Wave 우선 체크 (손바닥 흔들기)
        if self._check_wave(pts_px[LM_WRIST][0]):
            self._wrist_x_history.clear()
            return GestureResult(gesture="WAVE", confidence=0.9)
        
        # 일반 제스처 분류
        gesture = self._classify_gesture(pts_px)
        
        out = GestureResult(gesture=gesture, confidence=0.8)
        
        # POINT → 짐벌 각도 계산
        if gesture == "POINT":
            z = estimate_depth(pts_px)
            direction = calc_pointing_vector(pts_px, z)
            pan, tilt = vector_to_pan_tilt(direction)
            out.pan_deg = pan
            out.tilt_deg = tilt
        
        # PINCH → 밝기 계산
        elif gesture == "PINCH":
            hand_size = hand_pixel_size(pts_px)
            if hand_size > 5:
                pinch_dist = np.linalg.norm(
                    pts_px[LM_THUMB_TIP][:2] - pts_px[LM_INDEX_TIP][:2]
                )
                ratio = pinch_dist / hand_size
                # 정규화: PINCH_BRIGHT_MIN ~ PINCH_BRIGHT_MAX_RATIO → 0~100%
                ratio_clamped = max(PINCH_BRIGHT_MIN_RATIO,
                                    min(PINCH_BRIGHT_MAX_RATIO, ratio))
                t = (ratio_clamped - PINCH_BRIGHT_MIN_RATIO) / \
                    (PINCH_BRIGHT_MAX_RATIO - PINCH_BRIGHT_MIN_RATIO)
                out.brightness = t * 100.0
        
        # 손 정지 감지 (Lock 전환용)
        out.confidence = 0.95 if self._check_hand_still(pts_px[LM_WRIST][:2]) else 0.7
        
        return out
    
    def shutdown(self):
        if self.hands is not None:
            self.hands.close()
        log.info("Vision pipeline shutdown")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown()