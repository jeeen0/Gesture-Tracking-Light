import cv2

from pi_runtime_config import FRAME_W


def hand_bbox_width_px(results):
    if not results or not getattr(results, "multi_hand_landmarks", None):
        return 0
    xs = [lm.x for lm in results.multi_hand_landmarks[0].landmark]
    return int((max(xs) - min(xs)) * FRAME_W)


def detect_wave_motion(prev_gray, gray, history):
    if prev_gray is None:
        return False, 0.0, 0

    diff = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    motion_score = cv2.countNonZero(thresh)
    if motion_score <= int(gray.shape[0] * gray.shape[1] * 0.0015):
        return False, 0.0, 0

    moments = cv2.moments(thresh)
    if moments["m00"] <= 0:
        return False, 0.0, 0

    history.append(int(moments["m10"] / moments["m00"]))
    if len(history) < 4:
        return False, 0.0, 0

    span = max(history) - min(history)
    steps = [history[i] - history[i - 1] for i in range(1, len(history))]
    signs = [1 if step > 1 else -1 for step in steps if abs(step) > 1]
    turns = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
    threshold = max(5, int(gray.shape[1] * 0.05))
    return span > threshold and turns >= 2, float(span), turns
