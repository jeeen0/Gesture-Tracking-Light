"""
pointing_target.py
──────────────────────────────────────────────────────────────────
라즈베리파이5 + RGB 와이드 카메라 전용
손가락이 실제로 가리키는 표면 지점을 화면에 표시

핵심 알고리즘: Ray Marching + MiDaS Depth Map
  1. MiDaS 모노큘러 depth 모델로 씬 전체 depth map 생성
  2. MediaPipe로 손가락 방향 Ray 구성
  3. Ray를 픽셀 단위로 따라가며 실제 표면(depth map)과
     첫 번째 충돌 지점을 타겟으로 확정

캘리브레이션:
  코드 상단 HAND_WRIST_TO_MIDDLE_MCP_M = 0.10  ← 손목(#0)~중지MCP(#9) 실측 길이(m)
  이 값만 넣으면 별도 동작 없이 즉시 적용됨 (R키 불필요)

설치:
  pip install mediapipe opencv-python numpy torch torchvision
  (MiDaS는 torch.hub로 자동 다운로드됨 - 첫 실행 시 ~수분 소요)

조작:
  q : 종료
  d : depth map 오버레이 토글
  r : 캘리브레이션 재측정 (손을 카메라 정면으로 펼치고 누름)
  스페이스 : 현재 타겟 좌표 출력
──────────────────────────────────────────────────────────────────
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import time
from collections import deque
from typing import Optional, Tuple

# ══════════════════════════════════════════════════════════════════
#  ★ 캘리브레이션 설정 - 여기만 수정하면 됨 ★
# ══════════════════════════════════════════════════════════════════
HAND_WRIST_TO_MIDDLE_MCP_M = 0.10
# ★ 손목뼈(landmark #0) ~ 중지 MCP(landmark #9) 실측 길이 (미터)
#
# [왜 #9인가? - #12(중지 끝)을 안 쓰는 이유]
#   손가락을 굽히면 #12 위치가 손목 쪽으로 크게 이동 → 픽셀 거리 줄어듦
#   → depth를 실제보다 크게 추정하는 오차 발생
#   #9(중지 MCP)는 손가락 어떻게 움직여도 위치가 거의 고정 → 안정적
#
# [측정 방법]
#   자로 손목 뼈 끝 ~ 손등 중지 첫 번째 볼록 관절까지 재서 입력
#   예: 10cm → 0.10 / 9cm → 0.09 / 11cm → 0.11
#   ★ 이 값만 맞게 넣으면 별도 동작(R키 등) 불필요. 코드 실행 즉시 적용됨.

# ══════════════════════════════════════════════════════════════════
#  카메라 설정
# ══════════════════════════════════════════════════════════════════
FRAME_W = 640   # 라즈베리파이 성능상 640x480 권장 (1280x720 시 느려짐)
FRAME_H = 480
CAM_INDEX = 0
MIRROR = True   # ★ True=좌우반전(셀피/미러 모드), False=원본

# ══════════════════════════════════════════════════════════════════
#  Ray Marching 설정
# ══════════════════════════════════════════════════════════════════
RAY_STEP_PX = 4          # Ray를 몇 픽셀씩 전진시킬지 (작을수록 정확, 느림)
RAY_MAX_STEPS = 300      # 최대 탐색 스텝 수
DEPTH_HIT_THRESHOLD = 0.08  # Ray depth가 실제 depth보다 이 비율만큼 가까워지면 충돌
                             # 0.05~0.15 사이에서 조정 (작으면 정밀, 크면 관대)
SMOOTHING_FRAMES = 5     # 타겟 좌표 스무딩 프레임 수 (떨림 방지)


class DepthEstimator:
    """MiDaS 모노큘러 depth 추정기"""

    def __init__(self):
        print("[Depth] MiDaS 모델 로딩 중... (첫 실행 시 다운로드 필요)")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # MiDaS small: 라즈베리파이에서 돌아가는 가장 가벼운 버전
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform

        print(f"[Depth] 모델 로딩 완료 ({self.device})")

        # depth scale 초기값 (캘리브레이션 전 임시값)
        self.depth_scale = 1.0   # MiDaS 상대 depth → 실제 미터 변환 계수
        self.is_calibrated = False

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        BGR 프레임 → depth map (float32, frame과 같은 H×W 크기)
        값이 클수록 가까움 (MiDaS 출력 특성)
        """
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(frame_bgr.shape[0], frame_bgr.shape[1]),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # 정규화: 0~1 (1=가깝다, 0=멀다)
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 1e-6:
            depth_map = (depth_map - d_min) / (d_max - d_min)

        return depth_map.astype(np.float32)

    def calibrate(self, depth_map: np.ndarray,
                  wrist_px: Tuple[int, int],
                  middle_mcp_px: Tuple[int, int],
                  cam_fx: float) -> bool:
        """
        손목(#0)~중지MCP(#9) 거리로 depth_scale 보정.

        원리:
          손목~중지 끝 픽셀 거리 + 카메라 fx → 손의 추정 depth 계산
          MiDaS depth값과 비교해 scale 도출
        """
        wx, wy = wrist_px
        mx, my = middle_mcp_px

        # 픽셀 거리
        px_dist = np.sqrt((mx - wx) ** 2 + (my - wy) ** 2)
        if px_dist < 20:
            return False  # 너무 가까이 있으면 캘리브 스킵

        # 핀홀 카메라 모델: real_size = depth * pixel_size / fx
        # → depth = real_size * fx / pixel_size
        estimated_depth_m = HAND_WRIST_TO_MIDDLE_MCP_M * cam_fx / px_dist

        # 손목 주변 depth map 평균값 (MiDaS 상대값)
        h, w = depth_map.shape
        wx_c = np.clip(wx, 5, w - 5)
        wy_c = np.clip(wy, 5, h - 5)
        midas_val = float(np.mean(depth_map[wy_c - 5:wy_c + 5, wx_c - 5:wx_c + 5]))

        if midas_val < 0.01:
            return False

        # MiDaS는 값이 클수록 가깝다 → 실제 거리 = scale / midas_val
        # estimated_depth_m = scale / midas_val
        self.depth_scale = estimated_depth_m * midas_val
        self.is_calibrated = True
        print(f"[Calib] depth_scale = {self.depth_scale:.4f}  "
              f"(손 depth ≈ {estimated_depth_m:.2f}m, MiDaS val = {midas_val:.3f})")
        return True

    def midas_to_meters(self, midas_val: float) -> float:
        """MiDaS 정규화값 → 실제 거리(m) 변환"""
        if midas_val < 0.001:
            return 99.0  # 매우 먼 곳
        return self.depth_scale / midas_val


class HandPointer:
    """MediaPipe 손 감지 + 검지 방향 벡터 추출"""

    # MediaPipe landmark 인덱스
    WRIST = 0
    INDEX_MCP = 5    # 검지 손허리뼈-손가락뼈 관절
    INDEX_TIP = 8    # 검지 끝
    MIDDLE_MCP = 9   # 중지 MCP 관절 (캘리브 기준 - 손가락 굽혀도 위치 안 변함)

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame_bgr: np.ndarray):
        """
        반환: (landmarks, result) 또는 (None, None)
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None, None
        return result.multi_hand_landmarks[0], result

    def get_key_points(self, lms, frame_w: int, frame_h: int) -> dict:
        """픽셀 좌표로 변환된 주요 랜드마크 반환"""
        def to_px(lm):
            return (int(lm.x * frame_w), int(lm.y * frame_h))

        return {
            "wrist": to_px(lms.landmark[self.WRIST]),
            "index_mcp": to_px(lms.landmark[self.INDEX_MCP]),
            "index_tip": to_px(lms.landmark[self.INDEX_TIP]),
            "middle_mcp": to_px(lms.landmark[self.MIDDLE_MCP]),
        }

    def is_pointing(self, lms) -> bool:
        """
        포인팅 제스처 판별:
          검지만 펴져 있고 나머지 손가락은 접힌 상태
        """
        lm = lms.landmark

        def finger_extended(tip, pip):
            return lm[tip].y < lm[pip].y  # 화면 y는 위가 0

        index_up = finger_extended(8, 6)
        middle_down = not finger_extended(12, 10)
        ring_down = not finger_extended(16, 14)
        pinky_down = not finger_extended(20, 18)

        return index_up and middle_down and ring_down and pinky_down

    def draw_landmarks(self, frame: np.ndarray, lms, result):
        self.mp_draw.draw_landmarks(
            frame, lms, self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(100, 220, 255), thickness=2, circle_radius=4),
            self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1),
        )


class RayMarcher:
    """
    손가락 방향 Ray를 depth map 위에서 전진시켜
    표면과 충돌하는 지점을 찾음
    """

    def __init__(self, frame_w: int, frame_h: int):
        self.w = frame_w
        self.h = frame_h

    def march(self,
              mcp_px: Tuple[int, int],
              tip_px: Tuple[int, int],
              depth_map: np.ndarray,
              depth_estimator: DepthEstimator) -> Optional[Tuple[int, int]]:
        """
        Ray 시작: MCP (손허리뼈 관절)
        Ray 방향: TIP - MCP 방향으로 연장

        depth_map 위에서 Ray를 RAY_STEP_PX씩 전진하면서:
          - Ray 현재 위치의 실제 depth (from depth_map)
          - Ray가 3D 공간에서 진행한 예상 depth
        를 비교. 예상 depth ≥ 실제 depth 이면 표면에 도달한 것.

        반환: 충돌 픽셀 좌표 또는 None
        """
        mx, my = mcp_px
        tx, ty = tip_px

        # 방향 벡터 (픽셀 공간)
        dx = tx - mx
        dy = ty - my
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < 1e-3:
            return None

        # 단위 벡터
        ux = dx / dist
        uy = dy / dist

        # MCP에서의 depth를 시작 depth로 사용
        mx_c = int(np.clip(mx, 0, self.w - 1))
        my_c = int(np.clip(my, 0, self.h - 1))
        start_midas = float(depth_map[my_c, mx_c])
        start_depth_m = depth_estimator.midas_to_meters(start_midas)

        # Ray를 TIP에서 시작해서 연장 방향으로 marching
        # (TIP 이전 구간은 손 자체이므로 스킵)
        # 시작 오프셋: tip까지의 픽셀 거리 + 약간 여유
        start_offset = dist + 10

        for step in range(RAY_MAX_STEPS):
            # 현재 픽셀 위치
            t = start_offset + step * RAY_STEP_PX
            cx = int(mx + ux * t)
            cy = int(my + uy * t)

            # 화면 밖이면 종료
            if cx < 0 or cx >= self.w or cy < 0 or cy >= self.h:
                return None

            # 현재 픽셀의 실제 depth (MiDaS)
            surface_midas = float(depth_map[cy, cx])
            surface_depth_m = depth_estimator.midas_to_meters(surface_midas)

            # Ray의 예상 depth: 시작 depth에서 전진 거리만큼 증가
            # (카메라 방향 성분만 고려한 근사값)
            # 3D Ray depth 계산: Δdepth ≈ Δpixel * start_depth / fx
            # 여기서는 단순히 픽셀 전진 비율로 추정
            ray_depth_m = start_depth_m * (t / dist) if dist > 0 else start_depth_m

            # ─── 충돌 판정 ───────────────────────────────────────────────
            # 표면이 Ray보다 가깝거나 같으면 (= Ray가 표면에 도달)
            # MiDaS: 클수록 가까움 → surface_midas > threshold_midas
            # 실거리로 비교: ray_depth_m >= surface_depth_m
            if ray_depth_m >= surface_depth_m * (1.0 - DEPTH_HIT_THRESHOLD):
                return (cx, cy)

        return None  # 타겟 못 찾음 (허공을 가리키거나 범위 초과)


class MotorAngleTracker:
    """
    3초 정지 감지 + 평균 타겟 + 팬틸트 각도 자동 출력

    동작 원리:
      - 매 프레임 raw 타겟 픽셀을 수집
      - 수집된 픽셀들의 표준편차가 STABLE_STD_PX 이하면 "정지 상태"로 판단
      - 정지 상태가 STABLE_SECONDS 초 이상 지속되면 평균 타겟 확정 → 각도 출력
      - 손이 많이 움직이면 (std > JITTER_RESET_PX) 버퍼 리셋

    팬틸트 각도 공식 (깊이 불필요):
      pan_deg  = (tx - cx) / fx * (180/π)   ← 좌우 각도
      tilt_deg = (ty - cy) / fy * (180/π)   ← 상하 각도
      (카메라 정면=0°, 오른쪽/아래 = 양수)
    """

    # ── 파라미터 (위에서 조정) ─────────────────────────────────────
    STABLE_SECONDS  = 3.0    # 몇 초 정지해야 확정할지
    STABLE_STD_PX   = 20     # 픽셀 표준편차 이 이하면 "정지" 판단
    JITTER_RESET_PX = 60     # 이 이상 흔들리면 버퍼 초기화
    BUFFER_MAXLEN   = 90     # 최대 90프레임(≈3초@30fps) 버퍼
    EMA_ALPHA       = 0.25   # 표시용 EMA 스무딩 계수

    def __init__(self, cam_fx: float, cam_fy: float,
                 cx: float, cy: float):
        self.fx = cam_fx
        self.fy = cam_fy
        self.cx = cx   # 화면 중심 x
        self.cy = cy   # 화면 중심 y

        self.buf_x: deque = deque(maxlen=self.BUFFER_MAXLEN)
        self.buf_y: deque = deque(maxlen=self.BUFFER_MAXLEN)
        self.stable_start: Optional[float] = None
        self.confirmed_target: Optional[Tuple[int, int]] = None
        self.confirmed_angles: Optional[Tuple[float, float]] = None

        # 표시용 EMA (실시간 시각화)
        self.ema_x: Optional[float] = None
        self.ema_y: Optional[float] = None

    def update(self, raw: Optional[Tuple[int, int]],
               is_pointing: bool) -> dict:
        """
        3초간 포인팅 유지 → 무조건 평균값 확정 출력 (흔들림 무관)
        """
        result = dict(
            display_target=None,
            confirmed=self.confirmed_target,
            pan_deg=self.confirmed_angles[0] if self.confirmed_angles else None,
            tilt_deg=self.confirmed_angles[1] if self.confirmed_angles else None,
            stable_ratio=0.0,
            std_px=0.0,
        )

        if not is_pointing or raw is None:
            self._reset_buffer()
            return result

        rx, ry = raw
        self.buf_x.append(rx)
        self.buf_y.append(ry)

        # EMA (표시용 스무딩)
        if self.ema_x is None:
            self.ema_x, self.ema_y = float(rx), float(ry)
        else:
            self.ema_x = self.EMA_ALPHA * rx + (1 - self.EMA_ALPHA) * self.ema_x
            self.ema_y = self.EMA_ALPHA * ry + (1 - self.EMA_ALPHA) * self.ema_y
        result["display_target"] = (int(self.ema_x), int(self.ema_y))

        # 타이머 시작
        if self.stable_start is None:
            self.stable_start = time.time()

        elapsed = time.time() - self.stable_start
        ratio = min(elapsed / self.STABLE_SECONDS, 1.0)
        result["stable_ratio"] = ratio

        # std는 참고용으로만
        if len(self.buf_x) >= 5:
            std = float(np.sqrt(np.std(self.buf_x)**2 + np.std(self.buf_y)**2))
            result["std_px"] = std

        # ── 3초 경과 → 무조건 확정 ──────────────────────────────
        if elapsed >= self.STABLE_SECONDS:
            tx = int(np.mean(self.buf_x))
            ty = int(np.mean(self.buf_y))
            self.confirmed_target = (tx, ty)
            pan, tilt = self._to_angles(tx, ty)
            self.confirmed_angles = (pan, tilt)
            result["confirmed"] = (tx, ty)
            result["pan_deg"] = pan
            result["tilt_deg"] = tilt
            result["stable_ratio"] = 1.0
            self._announce(tx, ty, pan, tilt, result["std_px"])
            self._reset_buffer()

        return result

    def _to_angles(self, tx: int, ty: int) -> Tuple[float, float]:
        """
        픽셀 → 팬틸트 각도 (도)

        핀홀 카메라 역투영:
          X_cam = (tx - cx) / fx   (무차원 방향 벡터 x성분)
          Y_cam = (ty - cy) / fy   (무차원 방향 벡터 y성분)
          pan   = atan(X_cam) → 도
          tilt  = atan(Y_cam) → 도

        깊이 불필요: 방향만 알면 됨.
        카메라 정면=0°, 오른쪽=+pan, 아래=+tilt
        """
        import math
        x_cam = (tx - self.cx) / self.fx
        y_cam = (ty - self.cy) / self.fy
        pan_deg  = math.degrees(math.atan(x_cam))
        tilt_deg = math.degrees(math.atan(y_cam))
        return pan_deg, tilt_deg

    def _announce(self, tx, ty, pan, tilt, std):
        print("\n" + "="*50)
        print(f"  ★ 타겟 확정!")
        print(f"  픽셀      : ({tx}, {ty})")
        print(f"  Pan  각도 : {pan:+.2f}°  (카메라 기준 좌우)")
        print(f"  Tilt 각도 : {tilt:+.2f}°  (카메라 기준 상하)")
        print(f"  흔들림 std: {std:.1f}px")
        print(f"  → 모터 명령: pan={pan:+.2f}°, tilt={tilt:+.2f}°")
        print("="*50)

    def _reset_buffer(self):
        self.buf_x.clear()
        self.buf_y.clear()
        self.stable_start = None
        self.ema_x = None
        self.ema_y = None


def draw_target(frame: np.ndarray, target: Tuple[int, int],
                is_confirmed: bool, stable_ratio: float = 0.0):
    """타겟 시각화 - 확정/탐색 상태 구분"""
    tx, ty = target

    if is_confirmed:
        # 확정 타겟: 초록 실선
        color = (0, 255, 80)
        cv2.circle(frame, (tx, ty), 24, color, 2)
        cv2.circle(frame, (tx, ty), 8, color, -1)
        cv2.line(frame, (tx - 32, ty), (tx + 32, ty), color, 2)
        cv2.line(frame, (tx, ty - 32), (tx, ty + 32), color, 2)
        cv2.putText(frame, "TARGET", (tx + 14, ty - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        # 수집 중: 노란 점선 + 진행 아크
        color = (0, 200, 255)
        cv2.circle(frame, (tx, ty), 24, color, 1)
        cv2.circle(frame, (tx, ty), 5, color, -1)
        # 진행률 아크 (안정화 progress)
        if stable_ratio > 0:
            angle = int(360 * stable_ratio)
            cv2.ellipse(frame, (tx, ty), (30, 30), -90, 0, angle, (0, 255, 200), 2)
        cv2.putText(frame, f"{int(stable_ratio*100)}%", (tx + 14, ty - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_ray_line(frame: np.ndarray,
                  mcp_px: Tuple[int, int],
                  tip_px: Tuple[int, int],
                  target: Optional[Tuple[int, int]]):
    """손가락에서 타겟까지 Ray 시각화"""
    end = target if target else tip_px

    # MCP → TIP: 흰색 실선
    cv2.line(frame, mcp_px, tip_px, (255, 255, 255), 2)
    # TIP → Target: 노란색 점선 느낌 (얇은 선)
    cv2.line(frame, tip_px, end, (0, 220, 255), 1)

    # TIP 점
    cv2.circle(frame, tip_px, 6, (255, 200, 0), -1)


def draw_hud(frame: np.ndarray, fps: float,
             is_calibrated: bool, is_pointing: bool,
             depth_overlay: bool,
             tracker_result: dict,
             depth_scale: float):
    """HUD - 팬틸트 각도 + 안정화 상태 표시"""
    # 캘리브 상태
    status_color = (0, 255, 100) if is_calibrated else (0, 160, 255)
    calib_text = f"CALIB OK  scale={depth_scale:.3f}" if is_calibrated else "CALIB NEEDED  (press R)"
    cv2.putText(frame, calib_text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    # 포인팅 상태
    pointing_text = "POINTING" if is_pointing else "no gesture"
    pointing_color = (0, 255, 255) if is_pointing else (120, 120, 120)
    cv2.putText(frame, pointing_text, (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, pointing_color, 1)

    # FPS + 흔들림
    std = tracker_result.get("std_px", 0.0)
    cv2.putText(frame, f"FPS {fps:.1f}  std={std:.1f}px",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # 안정화 진행 텍스트
    ratio = tracker_result.get("stable_ratio", 0.0)
    if ratio > 0 and ratio < 1.0:
        pct = int(ratio * 100)
        cv2.putText(frame, f"안정화 중... {pct}%  ({pct*3//100}s)",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    # 확정 타겟 각도
    pan  = tracker_result.get("pan_deg")
    tilt = tracker_result.get("tilt_deg")
    conf = tracker_result.get("confirmed")
    if conf and pan is not None:
        cv2.putText(frame, f"Pan {pan:+.1f}deg  Tilt {tilt:+.1f}deg",
                    (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2)
        cv2.putText(frame, f"pixel ({conf[0]}, {conf[1]})",
                    (10, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 80), 1)

    hint = "[Q]quit  [D]depth  [R]recalib  [SPC]강제출력"
    cv2.putText(frame, hint, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)


def main():
    print("=" * 55)
    print("  Pointing → Motor Angle Detector")
    print(f"  손목~중지MCP: {HAND_WRIST_TO_MIDDLE_MCP_M*100:.0f}cm")
    print(f"  안정화 시간: {MotorAngleTracker.STABLE_SECONDS}초 / "
          f"허용 흔들림: ±{MotorAngleTracker.STABLE_STD_PX}px")
    print("  조작: Q=종료 / D=depth 오버레이 / R=캘리브 / SPC=강제출력")
    print("=" * 55)

    depth_est   = DepthEstimator()
    hand_pointer = HandPointer()
    ray_marcher  = RayMarcher(FRAME_W, FRAME_H)

    cam_fx = FRAME_W * 0.7   # 와이드 렌즈 근사
    cam_fy = cam_fx
    cx, cy = FRAME_W / 2.0, FRAME_H / 2.0

    tracker = MotorAngleTracker(cam_fx, cam_fy, cx, cy)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    show_depth    = False
    fps_times: deque = deque(maxlen=30)
    prev_time = time.time()
    DEPTH_UPDATE_INTERVAL = 3
    frame_count = 0
    depth_map   = None

    print("\n손을 보이게 하고 검지로 3초간 가리키면 자동으로 각도가 출력됩니다.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] 카메라 읽기 실패")
            break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        if MIRROR:
            frame = cv2.flip(frame, 1)
        display = frame.copy()
        frame_count += 1

        # FPS
        now = time.time()
        fps_times.append(now - prev_time)
        prev_time = now
        fps = 1.0 / (sum(fps_times) / len(fps_times) + 1e-9)

        # Depth Map
        if frame_count % DEPTH_UPDATE_INTERVAL == 0:
            depth_map = depth_est.estimate(frame)

        # Depth 오버레이
        if show_depth and depth_map is not None:
            depth_vis = (depth_map * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
            display = cv2.addWeighted(display, 0.55, depth_colored, 0.45, 0)

        # 손 감지
        lms, mp_result = hand_pointer.process(frame)
        is_pointing_now = False
        raw_target = None

        if lms is not None:
            hand_pointer.draw_landmarks(display, lms, mp_result)
            pts = hand_pointer.get_key_points(lms, FRAME_W, FRAME_H)
            is_pointing_now = hand_pointer.is_pointing(lms)

            if depth_map is not None:
                if not depth_est.is_calibrated:
                    depth_est.calibrate(
                        depth_map, pts["wrist"], pts["middle_mcp"], cam_fx
                    )
                if is_pointing_now:
                    raw_target = ray_marcher.march(
                        mcp_px=pts["index_mcp"],
                        tip_px=pts["index_tip"],
                        depth_map=depth_map,
                        depth_estimator=depth_est,
                    )
                    if raw_target:
                        draw_ray_line(display, pts["index_mcp"],
                                      pts["index_tip"], raw_target)

        # ── 안정화 트래커 업데이트 ────────────────────────────────
        tr = tracker.update(raw_target, is_pointing_now)

        # 타겟 표시
        disp = tr["display_target"]
        conf = tr["confirmed"]
        if disp:
            draw_target(display, disp,
                        is_confirmed=(conf is not None),
                        stable_ratio=tr["stable_ratio"])

        # HUD
        draw_hud(display, fps,
                 depth_est.is_calibrated, is_pointing_now,
                 show_depth, tr, depth_est.depth_scale)

        cv2.imshow("Pointing Target Detector", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('d'):
            show_depth = not show_depth
            print(f"[D] Depth overlay: {'ON' if show_depth else 'OFF'}")

        elif key == ord('r'):
            print("\n[R] 캘리브레이션 모드")
            print("    손을 카메라 정면에 펼치고 아무 키나 누르세요...")
            while True:
                ret2, cframe = cap.read()
                if not ret2:
                    break
                cframe = cv2.resize(cframe, (FRAME_W, FRAME_H))
                if MIRROR:
                    cframe = cv2.flip(cframe, 1)
                cv2.putText(cframe,
                            "CALIB: Hold hand flat toward camera, press any key",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
                cv2.imshow("Pointing Target Detector", cframe)
                k2 = cv2.waitKey(30) & 0xFF
                if k2 != 255:
                    d_map = depth_est.estimate(cframe)
                    lms2, _ = hand_pointer.process(cframe)
                    if lms2:
                        pts2 = hand_pointer.get_key_points(lms2, FRAME_W, FRAME_H)
                        ok = depth_est.calibrate(
                            d_map, pts2["wrist"], pts2["middle_mcp"], cam_fx
                        )
                        print(f"    {'✓ 성공' if ok else '✗ 실패 - 다시 시도'}")
                    else:
                        print("    ✗ 손 미감지")
                    break

        elif key == ord(' '):
            # 강제 출력 (3초 기다리지 않고 현재 평균 즉시 출력)
            if disp:
                tx, ty = disp
                import math
                pan  = math.degrees(math.atan((tx - cx) / cam_fx))
                tilt = math.degrees(math.atan((ty - cy) / cam_fy))
                print(f"\n[SPC] 강제 출력 (현재 EMA 위치)")
                print(f"  픽셀: ({tx}, {ty})")
                print(f"  Pan : {pan:+.2f}°")
                print(f"  Tilt: {tilt:+.2f}°")
            else:
                print("[SPC] 타겟 없음 - 검지로 포인팅하세요")

    cap.release()
    cv2.destroyAllWindows()
    print("종료.")


if __name__ == "__main__":
    main()