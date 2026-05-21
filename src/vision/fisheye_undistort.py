"""
Fisheye 카메라 캘리브레이션 적용 — 손가락 추적 전 프레임 왜곡 보정.

사용 흐름:
    undist = FisheyeUndistorter("src/calibration/fisheye_undistort_map.npz",
                                target_size=(FRAME_W, FRAME_H))
    frame = undist.undistort(frame)
    # undist.fx, undist.fy, undist.cx, undist.cy 는 rectified 좌표계의 실제 intrinsics
"""
import cv2
import numpy as np


class FisheyeUndistorter:
    """캘리브레이션된 fisheye → rectified 변환. 런타임 해상도에 맞춰 맵을 재생성한다."""

    def __init__(self, npz_path, target_size):
        """
        npz_path: 캘리브레이션 파일 경로 (.npz)
        target_size: 런타임 프레임 (W, H)
        """
        data = np.load(npz_path)
        K_orig = data["camera_matrix"]      # (3, 3)
        D = data["dist_coeffs"]             # (4, 1)
        P_orig = data["new_camera_matrix"]  # (3, 3)
        calib_w, calib_h = (int(x) for x in data["image_size"])

        tw, th = int(target_size[0]), int(target_size[1])
        sx = tw / calib_w
        sy = th / calib_h

        # K (원본 카메라 행렬)와 P (rectified 카메라 행렬) 모두 target 해상도로 스케일
        K = K_orig.copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy

        P = P_orig.copy()
        P[0, 0] *= sx
        P[1, 1] *= sy
        P[0, 2] *= sx
        P[1, 2] *= sy

        # target_size에 맞춘 remap 맵 생성 (한 번만 수행)
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), P, (tw, th), cv2.CV_16SC2,
        )
        self.K = K
        self.D = D
        self.P = P
        self.target_size = (tw, th)
        self.calib_size = (calib_w, calib_h)
        self.rms = float(data["rms"])

    def undistort(self, frame):
        """fisheye 보정된 frame 반환. 입력과 동일 해상도."""
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

    # rectified intrinsics — pan/tilt 계산에 사용
    @property
    def fx(self):
        return float(self.P[0, 0])

    @property
    def fy(self):
        return float(self.P[1, 1])

    @property
    def cx(self):
        return float(self.P[0, 2])

    @property
    def cy(self):
        return float(self.P[1, 2])
