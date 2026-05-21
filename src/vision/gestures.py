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

        self.FOUR_FINGER_AIM_Z_THRESHOLD = 0.002
        self.FOUR_FINGER_AIM_PROJECTION_RATIO_MAX = 0.98
        self.FOUR_FINGER_OPEN_3D_THRESHOLD = 0.45
        self.FOUR_FINGER_SWIPE_THRESHOLD = 0.06
        self.FOUR_FINGER_SWIPE_Y_MAX = 0.35
        self.FOUR_FINGER_SWIPE_PALM_MAX = 0.35
        self.FOUR_FINGER_SWIPE_MIN_TIME = 0.05
        self.FOUR_FINGER_SWIPE_MAX_TIME = 0.45
        self.FOUR_FINGER_PALM_STABILITY_MAX = 0.70
        self.FOUR_FINGER_SWIPE_TO_PALM_RATIO = 2.0
        self.FOUR_FINGER_HORIZONTAL_RATIO = 1.2
        self.FOUR_FINGER_MAX_DIRECTION_TURNS = 0
        self.FOUR_FINGER_SWIPE_HISTORY_FRAMES = 10
        self.TICK_COOLDOWN = 0.80
        self.MIRROR_TICK_DIRECTION = True
        self.THUMB_DIRECTION_THRESHOLD = 0.30
        self.THUMB_SMOOTHING_FRAMES = 6
        self.THUMB_SMOOTHING_MIN_HITS = 4

        self.wrist_history = deque()
        self.four_finger_swipe_history = deque(maxlen=self.FOUR_FINGER_SWIPE_HISTORY_FRAMES)
        self.thumb_gesture_history = deque(maxlen=self.THUMB_SMOOTHING_FRAMES)
        self.neutral_palm_x = None
        self.neutral_palm_y = None
        self.tick_state = "READY"
        self.tick_direction = None
        self.cooldown_start_time = 0.0
        self.debug_state = self._empty_debug_state()

    def _empty_debug_state(self):
        return {
            "point_mode_candidate": False,
            "four_finger_aim_candidate": False,
            "index_open": False,
            "middle_open": False,
            "ring_open": False,
            "pinky_open": False,
            "thumbs_up_candidate": False,
            "thumbs_down_candidate": False,
            "thumb_direction_delta": 0.0,
            "aim_vector_x": 0.0,
            "aim_vector_y": 0.0,
            "aim_vector_z": 0.0,
            "aim_depth_ok": False,
            "aim_projection_ratio": 1.0,
            "fist_blocked_by_four_finger_aim": False,
            "swipe_delta_x": 0.0,
            "swipe_delta_y": 0.0,
            "swipe_elapsed": 0.0,
            "swipe_turns": 0,
            "four_finger_reject_reason": "",
            "finger_offset_x": None,
            "relative_offset_x": 0.0,
            "finger_offset_y": None,
            "relative_offset_y": 0.0,
            "mapped_offset_x": 0.0,
            "palm_center_x": None,
            "palm_center_y": None,
            "palm_move": 0.0,
            "tick_state": self.tick_state,
            "tick_direction": self.tick_direction,
            "left_tick_detected": False,
            "right_tick_detected": False,
            "cooldown_remaining": 0.0,
            "confirmed_gesture": None,
        }

    def _get_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def _get_distance_3d(self, p1, p2):
        return math.sqrt(
            (p1.x - p2.x) ** 2 +
            (p1.y - p2.y) ** 2 +
            (getattr(p1, "z", 0.0) - getattr(p2, "z", 0.0)) ** 2
        )

    def _projection_ratio(self, p1, p2):
        distance_3d = self._get_distance_3d(p1, p2)
        if distance_3d <= 1e-6:
            return 1.0
        return self._get_distance(p1, p2) / distance_3d

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

    def _get_tick_palm_center_point(self, landmarks):
        return self._average_point([
            landmarks[self.WRIST],
            landmarks[self.INDEX_FINGER_MCP],
            landmarks[self.MIDDLE_FINGER_MCP],
        ])

    def _get_four_finger_base_center_point(self, landmarks):
        return self._average_point([
            landmarks[self.INDEX_FINGER_MCP],
            landmarks[self.MIDDLE_FINGER_MCP],
            landmarks[self.RING_FINGER_MCP],
            landmarks[self.PINKY_MCP],
        ])

    def _get_four_finger_tip_center_point(self, landmarks):
        return self._average_point([
            landmarks[self.INDEX_FINGER_TIP],
            landmarks[self.MIDDLE_FINGER_TIP],
            landmarks[self.RING_FINGER_TIP],
            landmarks[self.PINKY_TIP],
        ])

    def _reset_tick_state(self):
        self.four_finger_swipe_history.clear()
        self.neutral_palm_x = None
        self.neutral_palm_y = None
        self.tick_state = "READY"
        self.tick_direction = None
        self.cooldown_start_time = 0.0

    def _reset_temporal_state(self):
        self.last_gesture = None
        self.gesture_counter = 0
        self._reset_tick_state()
        self.thumb_gesture_history.clear()
        self.debug_state = self._empty_debug_state()

    def _update_four_finger_aim(self, current_time, palm_center, base_center, tip_center, palm_size):
        aim_vector_x = (tip_center.x - base_center.x) / palm_size
        aim_vector_y = (tip_center.y - base_center.y) / palm_size
        aim_vector_z = (tip_center.z - base_center.z) / palm_size
        aim_projection_ratio = self._projection_ratio(tip_center, base_center)
        aim_depth_ok = (
            abs(aim_vector_z) > self.FOUR_FINGER_AIM_Z_THRESHOLD or
            aim_projection_ratio < self.FOUR_FINGER_AIM_PROJECTION_RATIO_MAX
        )

        if self.neutral_palm_x is None:
            self.neutral_palm_x = palm_center.x
            self.neutral_palm_y = palm_center.y

        palm_move_from_start = math.sqrt(
            (palm_center.x - self.neutral_palm_x) ** 2 +
            (palm_center.y - self.neutral_palm_y) ** 2
        ) / palm_size
        cooldown_remaining = 0.0

        self.four_finger_swipe_history.append((
            current_time,
            tip_center.x - palm_center.x,
            tip_center.y - palm_center.y,
            palm_center.x,
            palm_center.y,
        ))

        swipe_delta_x = 0.0
        swipe_delta_y = 0.0
        swipe_elapsed = 0.0
        palm_delta = 0.0
        swipe_turns = 0
        current_tip_offset_x = tip_center.x - palm_center.x
        current_tip_offset_y = tip_center.y - palm_center.y
        for sample_time, sample_tip_offset_x, sample_tip_offset_y, sample_palm_x, sample_palm_y in self.four_finger_swipe_history:
            elapsed = current_time - sample_time
            if elapsed < self.FOUR_FINGER_SWIPE_MIN_TIME or elapsed > self.FOUR_FINGER_SWIPE_MAX_TIME:
                continue

            candidate_dx = (current_tip_offset_x - sample_tip_offset_x) / palm_size
            if abs(candidate_dx) <= abs(swipe_delta_x):
                continue

            swipe_delta_x = candidate_dx
            swipe_delta_y = (current_tip_offset_y - sample_tip_offset_y) / palm_size
            swipe_elapsed = elapsed
            palm_delta = math.sqrt(
                (palm_center.x - sample_palm_x) ** 2 +
                (palm_center.y - sample_palm_y) ** 2
            ) / palm_size

        offset_steps = [
            self.four_finger_swipe_history[i][1] - self.four_finger_swipe_history[i - 1][1]
            for i in range(1, len(self.four_finger_swipe_history))
        ]
        offset_signs = [
            1 if step > 0 else -1
            for step in offset_steps
            if abs(step) / palm_size > 0.01
        ]
        swipe_turns = sum(
            1
            for i in range(1, len(offset_signs))
            if offset_signs[i] != offset_signs[i - 1]
        )

        mapped_offset_x = -swipe_delta_x if self.MIRROR_TICK_DIRECTION else swipe_delta_x
        if self.tick_state == "COOLDOWN":
            cooldown_remaining = max(
                0.0,
                self.TICK_COOLDOWN - (current_time - self.cooldown_start_time),
            )

        self.debug_state.update({
            "four_finger_aim_candidate": True,
            "aim_vector_x": aim_vector_x,
            "aim_vector_y": aim_vector_y,
            "aim_vector_z": aim_vector_z,
            "aim_depth_ok": aim_depth_ok,
            "aim_projection_ratio": aim_projection_ratio,
            "finger_offset_x": aim_vector_x,
            "relative_offset_x": swipe_delta_x,
            "mapped_offset_x": mapped_offset_x,
            "finger_offset_y": aim_vector_y,
            "relative_offset_y": swipe_delta_y,
            "palm_center_x": palm_center.x,
            "palm_center_y": palm_center.y,
            "palm_move": palm_delta,
            "swipe_delta_x": swipe_delta_x,
            "swipe_delta_y": swipe_delta_y,
            "swipe_elapsed": swipe_elapsed,
            "swipe_turns": swipe_turns,
            "four_finger_reject_reason": "",
            "tick_state": self.tick_state,
            "tick_direction": self.tick_direction,
            "cooldown_remaining": cooldown_remaining,
        })

        def reject(reason):
            self.debug_state["four_finger_reject_reason"] = reason
            return None

        if self.tick_state != "READY":
            if cooldown_remaining <= 0:
                return reject("release_needed")
            return reject("cooldown")

        if not aim_depth_ok:
            return reject("depth")

        if palm_move_from_start >= self.FOUR_FINGER_PALM_STABILITY_MAX:
            return reject("palm_start")

        if palm_delta >= self.FOUR_FINGER_SWIPE_PALM_MAX:
            return reject("palm_delta")

        if abs(swipe_delta_y) >= self.FOUR_FINGER_SWIPE_Y_MAX:
            return reject("vertical")

        if swipe_turns > self.FOUR_FINGER_MAX_DIRECTION_TURNS:
            return reject("turns")

        if abs(swipe_delta_x) < palm_delta * self.FOUR_FINGER_SWIPE_TO_PALM_RATIO:
            return reject("palm_ratio")

        if abs(swipe_delta_x) < abs(swipe_delta_y) * self.FOUR_FINGER_HORIZONTAL_RATIO:
            return reject("horizontal")

        if mapped_offset_x > self.FOUR_FINGER_SWIPE_THRESHOLD:
            self.tick_state = "TRIGGERED_RIGHT"
            self.tick_direction = "RIGHT"
            self.cooldown_start_time = current_time
            self.four_finger_swipe_history.clear()
            self.debug_state["tick_state"] = "TRIGGERED_RIGHT"
            self.debug_state["tick_direction"] = self.tick_direction
            self.debug_state["right_tick_detected"] = True
            self.debug_state["cooldown_remaining"] = self.TICK_COOLDOWN
            self.tick_state = "COOLDOWN"
            return "FOUR_FINGER_AIM_RIGHT"

        if mapped_offset_x < -self.FOUR_FINGER_SWIPE_THRESHOLD:
            self.tick_state = "TRIGGERED_LEFT"
            self.tick_direction = "LEFT"
            self.cooldown_start_time = current_time
            self.four_finger_swipe_history.clear()
            self.debug_state["tick_state"] = "TRIGGERED_LEFT"
            self.debug_state["tick_direction"] = self.tick_direction
            self.debug_state["left_tick_detected"] = True
            self.debug_state["cooldown_remaining"] = self.TICK_COOLDOWN
            self.tick_state = "COOLDOWN"
            return "FOUR_FINGER_AIM_LEFT"

        return reject("threshold")

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

        dist_index = self._get_distance(landmarks[self.INDEX_FINGER_TIP], landmarks[self.WRIST])
        dist_middle = self._get_distance(landmarks[self.MIDDLE_FINGER_TIP], landmarks[self.WRIST])
        dist_ring = self._get_distance(landmarks[self.RING_FINGER_TIP], landmarks[self.WRIST])
        dist_pinky = self._get_distance(landmarks[self.PINKY_TIP], landmarks[self.WRIST])

        wrist_to_index_mcp = self._get_distance(landmarks[self.INDEX_FINGER_MCP], landmarks[self.WRIST])
        wrist_to_middle_mcp = self._get_distance(landmarks[self.MIDDLE_FINGER_MCP], landmarks[self.WRIST])

        index_open = dist_index > wrist_to_index_mcp * 1.2
        middle_open = dist_middle > wrist_to_middle_mcp * 1.25
        ring_open = dist_ring > wrist_to_middle_mcp * 1.25
        pinky_open = dist_pinky > wrist_to_middle_mcp * 1.15

        self.debug_state.update({
            "index_open": index_open,
            "middle_open": middle_open,
            "ring_open": ring_open,
            "pinky_open": pinky_open,
        })

        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_mcp = landmarks[self.THUMB_MCP]
        index_tip = landmarks[self.INDEX_FINGER_TIP]
        palm_center = self._get_palm_center_point(landmarks)
        palm_ref = self._get_distance(landmarks[self.MIDDLE_FINGER_MCP], landmarks[self.WRIST])
        palm_width = self._get_distance(
            landmarks[self.INDEX_FINGER_MCP],
            landmarks[self.PINKY_MCP],
        )
        palm_size = max(palm_ref, palm_width, 1e-6)

        thumb_palm_dist = self._get_distance(thumb_tip, palm_center)
        thumb_folded = thumb_palm_dist < palm_ref * 0.70
        thumb_tip_to_mcp = self._get_distance(thumb_tip, thumb_mcp)
        thumb_open = (
            not thumb_folded and
            thumb_tip_to_mcp > palm_size * 0.28
        )

        index_ratio = dist_index / max(wrist_to_index_mcp, 1e-6)
        middle_ratio = dist_middle / max(wrist_to_middle_mcp, 1e-6)
        ring_ratio = dist_ring / max(wrist_to_middle_mcp, 1e-6)
        pinky_ratio = dist_pinky / max(wrist_to_middle_mcp, 1e-6)
        avg_ratio = (index_ratio + middle_ratio + ring_ratio + pinky_ratio) / 4.0

        is_fist = (
            avg_ratio < 1.18 and
            middle_ratio < 1.20 and
            ring_ratio < 1.20 and
            pinky_ratio < 1.20 and
            index_ratio < 1.30 and
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
        index_aim_open = self._get_distance_3d(
            landmarks[self.INDEX_FINGER_TIP],
            landmarks[self.INDEX_FINGER_MCP],
        ) > palm_size * self.FOUR_FINGER_OPEN_3D_THRESHOLD
        middle_aim_open = self._get_distance_3d(
            landmarks[self.MIDDLE_FINGER_TIP],
            landmarks[self.MIDDLE_FINGER_MCP],
        ) > palm_size * self.FOUR_FINGER_OPEN_3D_THRESHOLD
        ring_aim_open = self._get_distance_3d(
            landmarks[self.RING_FINGER_TIP],
            landmarks[self.RING_FINGER_MCP],
        ) > palm_size * self.FOUR_FINGER_OPEN_3D_THRESHOLD
        pinky_aim_open = self._get_distance_3d(
            landmarks[self.PINKY_TIP],
            landmarks[self.PINKY_MCP],
        ) > palm_size * self.FOUR_FINGER_OPEN_3D_THRESHOLD
        # POINT_MODE enters the point-control mode. The actual POINT gesture is
        # kept separate so it can be reassigned/tuned independently.
        point_mode_candidate = (
            thumb_open and
            index_open and
            not middle_open and
            not ring_open and
            pinky_open and
            not open_palm and
            not is_fist
        )
        # POINT should work at distance too, so avoid absolute screen-space
        # distance checks. The core shape is index open with the other fingers
        # folded; the thumb may be folded or relaxed.
        is_point = (
            index_open and
            other_three_folded and
            not open_palm and
            not is_fist
        )
        is_mode_switch = (
            not index_open and
            not middle_open and
            not ring_open and
            pinky_open
        )

        four_finger_base_center = self._get_four_finger_base_center_point(landmarks)
        four_finger_tip_center = self._get_four_finger_tip_center_point(landmarks)
        aim_vector_x = (four_finger_tip_center.x - four_finger_base_center.x) / palm_size
        aim_vector_y = (four_finger_tip_center.y - four_finger_base_center.y) / palm_size
        aim_vector_z = (four_finger_tip_center.z - four_finger_base_center.z) / palm_size
        aim_projection_ratio = self._projection_ratio(four_finger_tip_center, four_finger_base_center)
        aim_depth_ok = (
            abs(aim_vector_z) > self.FOUR_FINGER_AIM_Z_THRESHOLD or
            aim_projection_ratio < self.FOUR_FINGER_AIM_PROJECTION_RATIO_MAX
        )
        # FOUR_FINGER_AIM은 라파 환경에서 노이즈가 많아 비활성화 (rasp 동기)
        four_finger_aim_candidate = False
        fist_blocked_by_four_finger_aim = (
            is_fist and
            four_finger_aim_candidate and
            aim_depth_ok
        )
        self.debug_state.update({
            "point_mode_candidate": point_mode_candidate,
            "four_finger_aim_candidate": four_finger_aim_candidate,
            "aim_vector_x": aim_vector_x,
            "aim_vector_y": aim_vector_y,
            "aim_vector_z": aim_vector_z,
            "aim_depth_ok": aim_depth_ok,
            "aim_projection_ratio": aim_projection_ratio,
            "fist_blocked_by_four_finger_aim": fist_blocked_by_four_finger_aim,
            "thumbs_up_candidate": thumbs_up_candidate,
            "thumbs_down_candidate": thumbs_down_candidate,
            "thumb_direction_delta": thumb_direction_delta,
        })

        # Priority 1: FIST
        if is_fist and not fist_blocked_by_four_finger_aim:
            self._reset_tick_state()
            self.last_gesture = "FIST"
            return "FIST", None

        # Priority 2: POINT_MODE
        if point_mode_candidate:
            self._reset_tick_state()
            self.last_gesture = "POINT_MODE"
            return "POINT_MODE", None

        # Priority 3: THUMBS_UP / THUMBS_DOWN brightness control.
        if confirmed_thumb_gesture:
            self.last_gesture = confirmed_thumb_gesture
            return confirmed_thumb_gesture, None

        # Priority 4: WAVE shape. app.py still requires wave motion and standby.
        if open_palm:
            self._reset_tick_state()
            self.last_gesture = "WAVE"
            return "WAVE", None

        # Priority 5: POINT
        if is_point:
            self._reset_tick_state()
            self.last_gesture = "POINT"
            return "POINT", None

        # Priority 6: MODE_SWITCH
        if is_mode_switch:
            self._reset_tick_state()
            self.last_gesture = "MODE_SWITCH"
            return "MODE_SWITCH", None

        self._reset_tick_state()
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
