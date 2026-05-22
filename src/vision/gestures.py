import math
import time
from collections import deque


class GestureRecognizer:
    def __init__(self):
        self.WRIST = 0
        self.THUMB_MCP = 2
        self.THUMB_IP = 3
        self.THUMB_TIP = 4

        self.INDEX_FINGER_MCP = 5
        self.INDEX_FINGER_PIP = 6
        self.INDEX_FINGER_TIP = 8

        self.MIDDLE_FINGER_MCP = 9
        self.MIDDLE_FINGER_PIP = 10
        self.MIDDLE_FINGER_TIP = 12

        self.RING_FINGER_MCP = 13
        self.RING_FINGER_PIP = 14
        self.RING_FINGER_TIP = 16

        self.PINKY_MCP = 17
        self.PINKY_PIP = 18
        self.PINKY_TIP = 20

        self.last_gesture = None
        self.gesture_counter = 0
        self.threshold = 5

        # 손가락 펴짐 판정 임계값 (tip→손목 거리 / 자기 MCP→손목 거리)
        # 펴졌을 때 비율이 약 1.7~2.5, 굽혔을 때 약 1.0~1.3 → 1.3~1.5 사이가 분리점
        self.INDEX_OPEN_THRESHOLD = 1.28
        self.MIDDLE_OPEN_THRESHOLD = 1.35
        self.RING_OPEN_THRESHOLD = 1.35
        self.PINKY_OPEN_THRESHOLD = 1.12   # 새끼는 짧아서 더 낮게

        # FIST 판정 임계값 (자기 MCP 기준 ratio 상한)
        self.FIST_AVG_RATIO_MAX = 1.35
        self.FIST_FINGER_RATIO_MAX = 1.40

        self.THUMB_DIRECTION_THRESHOLD = 0.30
        self.THUMB_SMOOTHING_FRAMES = 6
        self.THUMB_SMOOTHING_MIN_HITS = 4

        self.wrist_history = deque()
        self.thumb_gesture_history = deque(maxlen=self.THUMB_SMOOTHING_FRAMES)
        self.debug_state = self._empty_debug_state()

    def _empty_debug_state(self):
        return {
            "point_mode_candidate": False,
            "index_open": False,
            "middle_open": False,
            "ring_open": False,
            "pinky_open": False,
            "thumbs_up_candidate": False,
            "thumbs_down_candidate": False,
            "thumb_direction_delta": 0.0,
            "confirmed_gesture": None,
        }

    def _get_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def _make_point(self, x, y, z=0.0):
        class P:
            pass

        p = P()
        p.x = x
        p.y = y
        p.z = z
        return p

    def _average_point(self, points):
        return self._make_point(
            sum(p.x for p in points) / len(points),
            sum(p.y for p in points) / len(points),
            sum(getattr(p, "z", 0.0) for p in points) / len(points),
        )

    def _get_palm_center_point(self, landmarks):
        return self._average_point([
            landmarks[self.WRIST],
            landmarks[self.INDEX_FINGER_MCP],
            landmarks[self.MIDDLE_FINGER_MCP],
            landmarks[self.RING_FINGER_MCP],
            landmarks[self.PINKY_MCP],
        ])

    def _reset_temporal_state(self):
        self.last_gesture = None
        self.gesture_counter = 0
        self.thumb_gesture_history.clear()
        self.debug_state = self._empty_debug_state()

    def extract_full_state(self, results):
        if not results or getattr(results, "multi_hand_landmarks", None) is None:
            self._reset_temporal_state()
            return {
                "hand_present": False,
                "gesture": None,
                "value": None,
                "wrist_x": None,
                "index_x": None,
                "servo_angle": 0,
                "move_dist": 0.0,
                **self.debug_state,
            }

        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark

        current_time = time.time()
        wrist = landmarks[self.WRIST]
        self.wrist_history.append((current_time, wrist.x, wrist.y))

        while self.wrist_history and current_time - self.wrist_history[0][0] > 3.0:
            self.wrist_history.popleft()

        move_dist = 0.0
        if len(self.wrist_history) > 10:
            xs = [x for _, x, _ in self.wrist_history]
            ys = [y for _, _, y in self.wrist_history]
            move_dist = (max(xs) - min(xs)) + (max(ys) - min(ys))

        gesture, value = self.recognize(landmarks, move_dist, current_time)
        self.debug_state["confirmed_gesture"] = gesture

        dx = landmarks[self.INDEX_FINGER_TIP].x - landmarks[self.WRIST].x
        dy = landmarks[self.INDEX_FINGER_TIP].y - landmarks[self.WRIST].y
        angle_rad = math.atan2(dx, -dy)
        servo_angle = int((math.degrees(angle_rad) + 360) % 360)

        return {
            "hand_present": True,
            "gesture": gesture,
            "value": value,
            "wrist_x": wrist.x,
            "index_x": landmarks[self.INDEX_FINGER_TIP].x,
            "servo_angle": servo_angle,
            "move_dist": move_dist,
            **self.debug_state,
        }

    def recognize(self, landmarks, move_dist, current_time=None):
        if current_time is None:
            current_time = time.time()

        self.debug_state = self._empty_debug_state()

        # 각 손가락 끝과 손목 거리
        dist_index = self._get_distance(landmarks[self.INDEX_FINGER_TIP], landmarks[self.WRIST])
        dist_middle = self._get_distance(landmarks[self.MIDDLE_FINGER_TIP], landmarks[self.WRIST])
        dist_ring = self._get_distance(landmarks[self.RING_FINGER_TIP], landmarks[self.WRIST])
        dist_pinky = self._get_distance(landmarks[self.PINKY_TIP], landmarks[self.WRIST])

        # 각 손가락 자기 MCP와 손목 거리 (정규화 기준)
        wrist_to_index_mcp = self._get_distance(landmarks[self.INDEX_FINGER_MCP], landmarks[self.WRIST])
        wrist_to_middle_mcp = self._get_distance(landmarks[self.MIDDLE_FINGER_MCP], landmarks[self.WRIST])
        wrist_to_ring_mcp = self._get_distance(landmarks[self.RING_FINGER_MCP], landmarks[self.WRIST])
        wrist_to_pinky_mcp = self._get_distance(landmarks[self.PINKY_MCP], landmarks[self.WRIST])

        # 손가락 펴짐 판정 (각각 자기 MCP 기준)
        index_open = dist_index > wrist_to_index_mcp * self.INDEX_OPEN_THRESHOLD
        middle_open = dist_middle > wrist_to_middle_mcp * self.MIDDLE_OPEN_THRESHOLD
        ring_open = dist_ring > wrist_to_ring_mcp * self.RING_OPEN_THRESHOLD
        pinky_open = dist_pinky > wrist_to_pinky_mcp * self.PINKY_OPEN_THRESHOLD

        self.debug_state.update({
            "index_open": index_open,
            "middle_open": middle_open,
            "ring_open": ring_open,
            "pinky_open": pinky_open,
        })

        # THUMB
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_mcp = landmarks[self.THUMB_MCP]
        palm_center = self._get_palm_center_point(landmarks)
        palm_ref = wrist_to_middle_mcp
        palm_width = self._get_distance(
            landmarks[self.INDEX_FINGER_MCP],
            landmarks[self.PINKY_MCP],
        )
        palm_size = max(palm_ref, palm_width, 1e-6)

        thumb_palm_dist = self._get_distance(thumb_tip, palm_center)
        thumb_folded = thumb_palm_dist < palm_ref * 0.55
        thumb_tip_to_mcp = self._get_distance(thumb_tip, thumb_mcp)
        thumb_open = (
            not thumb_folded and
            thumb_tip_to_mcp > palm_size * 0.20
        )

        # FIST 판정 (자기 MCP 기준 ratio가 모두 작음)
        index_ratio = dist_index / max(wrist_to_index_mcp, 1e-6)
        middle_ratio = dist_middle / max(wrist_to_middle_mcp, 1e-6)
        ring_ratio = dist_ring / max(wrist_to_ring_mcp, 1e-6)
        pinky_ratio = dist_pinky / max(wrist_to_pinky_mcp, 1e-6)
        avg_ratio = (index_ratio + middle_ratio + ring_ratio + pinky_ratio) / 4.0

        is_fist = (
            avg_ratio < self.FIST_AVG_RATIO_MAX and
            middle_ratio < self.FIST_FINGER_RATIO_MAX and
            ring_ratio < self.FIST_FINGER_RATIO_MAX and
            pinky_ratio < self.FIST_FINGER_RATIO_MAX and
            index_ratio < self.FIST_FINGER_RATIO_MAX and
            thumb_folded
        )

        other_three_folded = not middle_open and not ring_open and not pinky_open
        thumb_direction_delta = (thumb_tip.y - thumb_mcp.y) / palm_size
        thumb_direction_clear = abs(thumb_direction_delta) > self.THUMB_DIRECTION_THRESHOLD
        thumb_only = thumb_open and not index_open and not middle_open and not ring_open and not pinky_open
        thumbs_up_candidate = (
            thumb_only and
            thumb_direction_clear and
            thumb_tip.y < thumb_mcp.y
        )
        thumbs_down_candidate = (
            thumb_only and
            thumb_direction_clear and
            thumb_tip.y > thumb_mcp.y
        )
        raw_thumb_gesture = None
        if thumbs_up_candidate:
            raw_thumb_gesture = "THUMBS_UP"
        elif thumbs_down_candidate:
            raw_thumb_gesture = "THUMBS_DOWN"

        self.thumb_gesture_history.append(raw_thumb_gesture)
        confirmed_thumb_gesture = None
        for candidate in ("THUMBS_UP", "THUMBS_DOWN"):
            if sum(1 for item in self.thumb_gesture_history if item == candidate) >= self.THUMB_SMOOTHING_MIN_HITS:
                confirmed_thumb_gesture = candidate
                break

        open_palm = index_open and middle_open and ring_open and pinky_open

        # POINT_MODE: 검지 + 새끼 + 엄지 (rock 사인)
        point_mode_candidate = (
            thumb_open and
            index_open and
            not middle_open and
            not ring_open and
            pinky_open and
            not open_palm and
            not is_fist
        )

        # POINT: 검지만 (다른 셋 모두 접힘)
        is_point = (
            index_open and
            other_three_folded and
            not open_palm and
            not is_fist
        )

        # MODE_SWITCH: 새끼만
        is_mode_switch = (
            not index_open and
            not middle_open and
            not ring_open and
            pinky_open
        )

        self.debug_state.update({
            "point_mode_candidate": point_mode_candidate,
            "thumbs_up_candidate": thumbs_up_candidate,
            "thumbs_down_candidate": thumbs_down_candidate,
            "thumb_direction_delta": thumb_direction_delta,
        })

        # Priority 1: FIST
        if is_fist:
            self.last_gesture = "FIST"
            return "FIST", None

        # Priority 2: POINT_MODE
        if point_mode_candidate:
            self.last_gesture = "POINT_MODE"
            return "POINT_MODE", None

        # Priority 3: THUMBS_UP / THUMBS_DOWN brightness control.
        if confirmed_thumb_gesture:
            self.last_gesture = confirmed_thumb_gesture
            return confirmed_thumb_gesture, None

        # Priority 4: WAVE shape. app.py still requires wave motion and standby.
        if open_palm:
            self.last_gesture = "WAVE"
            return "WAVE", None

        # Priority 5: POINT
        if is_point:
            self.last_gesture = "POINT"
            return "POINT", None

        # Priority 6: MODE_SWITCH
        if is_mode_switch:
            self.last_gesture = "MODE_SWITCH"
            return "MODE_SWITCH", None

        self.last_gesture = None
        self.gesture_counter = 0
        return None, None

    def _debounce(self, gesture_name):
        if gesture_name == self.last_gesture:
            self.gesture_counter += 1
            if self.gesture_counter >= self.threshold:
                return gesture_name, None
        else:
            self.last_gesture = gesture_name
            self.gesture_counter = 1
        return None, None
