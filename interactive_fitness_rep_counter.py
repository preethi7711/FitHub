"""
Interactive multi-exercise rep counter engine using MediaPipe + OpenCV.

Install:
    pip install opencv-python mediapipe numpy

Run:
    python interactive_fitness_rep_counter.py
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Optional, Tuple
import time

import cv2
import mediapipe as mp
import numpy as np


# ----------------------------- Configuration ----------------------------- #
CAMERA_INDEX = 0
VISIBILITY_THRESHOLD = 0.5
SMOOTHING_WINDOW = 5
DOWN_HOLD_FRAMES = 3
CURL_ARM = "right"  # left | right
INITIAL_EXERCISE_MODE = "squat"


def calculate_angle(a: Tuple[float, float],
                    b: Tuple[float, float],
                    c: Tuple[float, float]) -> float:
    """Calculate angle ABC in degrees from 2D points."""
    a_np = np.array(a, dtype=np.float32)
    b_np = np.array(b, dtype=np.float32)
    c_np = np.array(c, dtype=np.float32)

    ba = a_np - b_np
    bc = c_np - b_np
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosine = float(np.dot(ba, bc) / denom)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _safe_landmark_xy(landmarks, idx: int) -> Optional[Tuple[float, float]]:
    if landmarks is None:
        return None
    lm = landmarks[idx]
    if hasattr(lm, "visibility") and lm.visibility < VISIBILITY_THRESHOLD:
        return None
    return float(lm.x), float(lm.y)


def _joint_angle(landmarks, a_idx: int, b_idx: int, c_idx: int) -> Optional[float]:
    a = _safe_landmark_xy(landmarks, a_idx)
    b = _safe_landmark_xy(landmarks, b_idx)
    c = _safe_landmark_xy(landmarks, c_idx)
    if a and b and c:
        return calculate_angle(a, b, c)
    return None


class ExerciseCounter(ABC):
    """Base exercise counter with smoothing + rep state machine."""

    def __init__(self,
                 name: str,
                 smoothing_window: int = SMOOTHING_WINDOW,
                 down_hold_frames: int = DOWN_HOLD_FRAMES):
        self.name = name
        self._smoothing_window = smoothing_window
        self._down_hold_frames = down_hold_frames
        self._angle_hist: Deque[float] = deque(maxlen=smoothing_window)
        self.reps = 0
        self.stage = "UP"
        self._rep_armed = False
        self._down_frames = 0

    def reset(self) -> None:
        self.reps = 0
        self.stage = "UP"
        self._rep_armed = False
        self._down_frames = 0
        self._angle_hist = deque(maxlen=self._smoothing_window)

    def _smooth(self, angle: Optional[float]) -> Optional[float]:
        if angle is None:
            return None
        self._angle_hist.append(float(angle))
        return float(np.mean(self._angle_hist))

    def _apply_standard_rep_logic(self, down_condition: bool, up_condition: bool) -> None:
        if down_condition:
            self._down_frames += 1
            if self._down_frames >= self._down_hold_frames and self.stage != "DOWN":
                self.stage = "DOWN"
                self._rep_armed = True
        else:
            if self.stage != "DOWN":
                self._down_frames = 0

        if self._rep_armed and up_condition:
            self.reps += 1
            self.stage = "UP"
            self._rep_armed = False
            self._down_frames = 0

    @abstractmethod
    def update(self, landmarks) -> Tuple[int, str, Optional[float]]:
        """Returns (reps, stage, angle)."""
        raise NotImplementedError


class SquatCounter(ExerciseCounter):
    def __init__(self):
        super().__init__(name="squat")

    def update(self, landmarks) -> Tuple[int, str, Optional[float]]:
        pose = mp.solutions.pose.PoseLandmark
        left_knee = _joint_angle(
            landmarks, pose.LEFT_HIP.value, pose.LEFT_KNEE.value, pose.LEFT_ANKLE.value
        )
        right_knee = _joint_angle(
            landmarks, pose.RIGHT_HIP.value, pose.RIGHT_KNEE.value, pose.RIGHT_ANKLE.value
        )
        vals = [v for v in [left_knee, right_knee] if v is not None]
        if not vals:
            return self.reps, self.stage, None

        angle = self._smooth(float(np.min(vals)))
        if angle is None:
            return self.reps, self.stage, None

        self._apply_standard_rep_logic(
            down_condition=angle <= 120.0,
            up_condition=angle >= 155.0,
        )
        return self.reps, self.stage, angle


class PushupCounter(ExerciseCounter):
    def __init__(self):
        super().__init__(name="pushup")

    def update(self, landmarks) -> Tuple[int, str, Optional[float]]:
        pose = mp.solutions.pose.PoseLandmark
        left_elbow = _joint_angle(
            landmarks, pose.LEFT_SHOULDER.value, pose.LEFT_ELBOW.value, pose.LEFT_WRIST.value
        )
        right_elbow = _joint_angle(
            landmarks, pose.RIGHT_SHOULDER.value, pose.RIGHT_ELBOW.value, pose.RIGHT_WRIST.value
        )
        vals = [v for v in [left_elbow, right_elbow] if v is not None]
        if not vals:
            return self.reps, self.stage, None

        angle = self._smooth(float(np.min(vals)))
        if angle is None:
            return self.reps, self.stage, None

        self._apply_standard_rep_logic(
            down_condition=angle <= 95.0,
            up_condition=angle >= 160.0,
        )
        return self.reps, self.stage, angle


class CurlCounter(ExerciseCounter):
    def __init__(self, arm: str = CURL_ARM):
        super().__init__(name="curl")
        self.arm = arm.lower()

    def update(self, landmarks) -> Tuple[int, str, Optional[float]]:
        pose = mp.solutions.pose.PoseLandmark
        if self.arm == "left":
            angle_raw = _joint_angle(
                landmarks, pose.LEFT_SHOULDER.value, pose.LEFT_ELBOW.value, pose.LEFT_WRIST.value
            )
        else:
            angle_raw = _joint_angle(
                landmarks, pose.RIGHT_SHOULDER.value, pose.RIGHT_ELBOW.value, pose.RIGHT_WRIST.value
            )

        angle = self._smooth(angle_raw)
        if angle is None:
            return self.reps, self.stage, None

        self._apply_standard_rep_logic(
            down_condition=angle <= 60.0,
            up_condition=angle >= 145.0,
        )
        return self.reps, self.stage, angle


class ShoulderPressCounter(ExerciseCounter):
    def __init__(self):
        super().__init__(name="shoulder_press")

    def update(self, landmarks) -> Tuple[int, str, Optional[float]]:
        pose = mp.solutions.pose.PoseLandmark
        left_elbow = _joint_angle(
            landmarks, pose.LEFT_SHOULDER.value, pose.LEFT_ELBOW.value, pose.LEFT_WRIST.value
        )
        right_elbow = _joint_angle(
            landmarks, pose.RIGHT_SHOULDER.value, pose.RIGHT_ELBOW.value, pose.RIGHT_WRIST.value
        )
        vals = [v for v in [left_elbow, right_elbow] if v is not None]
        if not vals:
            return self.reps, self.stage, None

        angle = self._smooth(float(np.max(vals)))
        if angle is None:
            return self.reps, self.stage, None

        self._apply_standard_rep_logic(
            down_condition=angle <= 100.0,
            up_condition=angle >= 160.0,
        )
        return self.reps, self.stage, angle


class LungeCounter(ExerciseCounter):
    def __init__(self):
        super().__init__(name="lunge")

    def update(self, landmarks) -> Tuple[int, str, Optional[float]]:
        pose = mp.solutions.pose.PoseLandmark
        left_knee = _joint_angle(
            landmarks, pose.LEFT_HIP.value, pose.LEFT_KNEE.value, pose.LEFT_ANKLE.value
        )
        right_knee = _joint_angle(
            landmarks, pose.RIGHT_HIP.value, pose.RIGHT_KNEE.value, pose.RIGHT_ANKLE.value
        )
        vals = [v for v in [left_knee, right_knee] if v is not None]
        if not vals:
            return self.reps, self.stage, None

        l_ankle = _safe_landmark_xy(landmarks, pose.LEFT_ANKLE.value)
        r_ankle = _safe_landmark_xy(landmarks, pose.RIGHT_ANKLE.value)
        if not (l_ankle and r_ankle):
            return self.reps, self.stage, None

        split_dist = abs(l_ankle[0] - r_ankle[0])
        angle = self._smooth(float(np.min(vals)))
        if angle is None:
            return self.reps, self.stage, None

        self._apply_standard_rep_logic(
            down_condition=split_dist >= 0.18 and angle <= 115.0,
            up_condition=angle >= 155.0,
        )
        return self.reps, self.stage, angle


class PlankCounter(ExerciseCounter):
    """Placeholder mode. No reps are counted; stage stays HOLD."""

    def __init__(self):
        super().__init__(name="plank")
        self.stage = "HOLD"

    def reset(self) -> None:
        self.reps = 0
        self.stage = "HOLD"
        self._rep_armed = False
        self._down_frames = 0
        self._angle_hist = deque(maxlen=self._smoothing_window)

    def update(self, landmarks) -> Tuple[int, str, Optional[float]]:
        pose = mp.solutions.pose.PoseLandmark
        left_body = _joint_angle(
            landmarks, pose.LEFT_SHOULDER.value, pose.LEFT_HIP.value, pose.LEFT_ANKLE.value
        )
        right_body = _joint_angle(
            landmarks, pose.RIGHT_SHOULDER.value, pose.RIGHT_HIP.value, pose.RIGHT_ANKLE.value
        )
        vals = [v for v in [left_body, right_body] if v is not None]
        if not vals:
            return self.reps, self.stage, None

        angle = self._smooth(float(np.mean(vals)))
        return self.reps, self.stage, angle


MODE_CONFIG = {
    ord("1"): ("squat", "Squat"),
    ord("2"): ("pushup", "Pushup"),
    ord("3"): ("curl", "Bicep Curl"),
    ord("4"): ("shoulder_press", "Shoulder Press"),
    ord("5"): ("lunge", "Lunge"),
    ord("6"): ("plank", "Plank"),
}


def build_counters() -> dict:
    return {
        "squat": SquatCounter(),
        "pushup": PushupCounter(),
        "curl": CurlCounter(),
        "shoulder_press": ShoulderPressCounter(),
        "lunge": LungeCounter(),
        "plank": PlankCounter(),
    }


def draw_overlay(frame,
                 mode_key: str,
                 mode_label: str,
                 reps: int,
                 stage: str,
                 angle: Optional[float],
                 fps: float) -> None:
    cv2.rectangle(frame, (10, 10), (430, 170), (0, 0, 0), -1)

    angle_text = "N/A" if angle is None else f"{angle:.1f}"
    cv2.putText(frame, f"Exercise: {mode_label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(frame, f"Reps: {reps}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Stage: {stage}", (20, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(frame, f"Angle: {angle_text}", (20, 141),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (280, 141),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    # Mode switch menu
    x0, y0 = 450, 10
    w, h = 270, 250
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), -1)
    cv2.putText(frame, "Press Keys to Switch:", (x0 + 12, y0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    menu_rows = [
        ("1", "Squat", "squat"),
        ("2", "Pushup", "pushup"),
        ("3", "Curl", "curl"),
        ("4", "Shoulder Press", "shoulder_press"),
        ("5", "Lunge", "lunge"),
        ("6", "Plank", "plank"),
    ]

    row_y = y0 + 58
    for key, label, mode in menu_rows:
        is_active = (mode == mode_key)
        color = (0, 255, 0) if is_active else (220, 220, 220)
        prefix = ">" if is_active else " "
        cv2.putText(frame, f"{prefix} {key} {label}", (x0 + 14, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        row_y += 30

    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)


def main() -> None:
    counters = build_counters()

    current_mode = INITIAL_EXERCISE_MODE
    if current_mode not in counters:
        current_mode = "squat"
    current_label = next((label for _, (m, label) in MODE_CONFIG.items() if m == current_mode), "Squat")
    counters[current_mode].reset()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Check device index/permissions.")

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    prev_time = time.time()

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(
                        color=(0, 255, 255), thickness=2, circle_radius=2
                    ),
                    connection_drawing_spec=mp_draw.DrawingSpec(
                        color=(255, 255, 255), thickness=2
                    ),
                )

            landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
            reps, stage, angle = counters[current_mode].update(landmarks)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-8)
            prev_time = now

            draw_overlay(frame, current_mode, current_label, reps, stage, angle, fps)
            cv2.imshow("Interactive Fitness Rep Counter", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key in MODE_CONFIG:
                next_mode, next_label = MODE_CONFIG[key]
                if next_mode != current_mode:
                    current_mode = next_mode
                    current_label = next_label
                    counters[current_mode].reset()
                    print(f"[SYSTEM] Switched to {current_label} Mode")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
