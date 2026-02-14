"""
Interactive fitness rep counter with Bird Flip mini-game (v2).

Install:
    pip install opencv-python mediapipe numpy

Run:
    python bird_flip_fitness_game_v2.py
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Optional, Tuple
import time
import math

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
GAME_MODE = True

# Bird physics tuning (balanced for rep-controlled gameplay)
GRAVITY = 0.4
FLAP_STRENGTH = 8.0
COUNTDOWN_SECONDS = 3
USE_COUNTDOWN = True


def calculate_angle(a: Tuple[float, float],
                    b: Tuple[float, float],
                    c: Tuple[float, float]) -> float:
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
    def __init__(self, name: str):
        self.name = name
        self.reps = 0
        self.stage = "UP"
        self._angle_hist: Deque[float] = deque(maxlen=SMOOTHING_WINDOW)
        self._rep_armed = False
        self._down_frames = 0

    def reset(self) -> None:
        self.reps = 0
        self.stage = "UP"
        self._angle_hist = deque(maxlen=SMOOTHING_WINDOW)
        self._rep_armed = False
        self._down_frames = 0

    def _smooth(self, angle: Optional[float]) -> Optional[float]:
        if angle is None:
            return None
        self._angle_hist.append(float(angle))
        return float(np.mean(self._angle_hist))

    def _apply_standard_rep_logic(self, down_condition: bool, up_condition: bool) -> None:
        if down_condition:
            self._down_frames += 1
            if self._down_frames >= DOWN_HOLD_FRAMES and self.stage != "DOWN":
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
        raise NotImplementedError


class SquatCounter(ExerciseCounter):
    def __init__(self):
        super().__init__("squat")

    def update(self, landmarks) -> Tuple[int, str, Optional[float]]:
        p = mp.solutions.pose.PoseLandmark
        left = _joint_angle(landmarks, p.LEFT_HIP.value, p.LEFT_KNEE.value, p.LEFT_ANKLE.value)
        right = _joint_angle(landmarks, p.RIGHT_HIP.value, p.RIGHT_KNEE.value, p.RIGHT_ANKLE.value)
        vals = [v for v in [left, right] if v is not None]
        if not vals:
            return self.reps, self.stage, None
        angle = self._smooth(float(np.min(vals)))
        if angle is None:
            return self.reps, self.stage, None
        self._apply_standard_rep_logic(angle <= 120.0, angle >= 155.0)
        return self.reps, self.stage, angle


class PushupCounter(ExerciseCounter):
    def __init__(self):
        super().__init__("pushup")

    def update(self, landmarks) -> Tuple[int, str, Optional[float]]:
        p = mp.solutions.pose.PoseLandmark
        left = _joint_angle(landmarks, p.LEFT_SHOULDER.value, p.LEFT_ELBOW.value, p.LEFT_WRIST.value)
        right = _joint_angle(landmarks, p.RIGHT_SHOULDER.value, p.RIGHT_ELBOW.value, p.RIGHT_WRIST.value)
        vals = [v for v in [left, right] if v is not None]
        if not vals:
            return self.reps, self.stage, None
        angle = self._smooth(float(np.min(vals)))
        if angle is None:
            return self.reps, self.stage, None
        self._apply_standard_rep_logic(angle <= 95.0, angle >= 160.0)
        return self.reps, self.stage, angle


class CurlCounter(ExerciseCounter):
    def __init__(self, arm: str = CURL_ARM):
        super().__init__("curl")
        self.arm = arm.lower()

    def update(self, landmarks) -> Tuple[int, str, Optional[float]]:
        p = mp.solutions.pose.PoseLandmark
        if self.arm == "left":
            raw = _joint_angle(landmarks, p.LEFT_SHOULDER.value, p.LEFT_ELBOW.value, p.LEFT_WRIST.value)
        else:
            raw = _joint_angle(landmarks, p.RIGHT_SHOULDER.value, p.RIGHT_ELBOW.value, p.RIGHT_WRIST.value)
        angle = self._smooth(raw)
        if angle is None:
            return self.reps, self.stage, None
        self._apply_standard_rep_logic(angle <= 60.0, angle >= 145.0)
        return self.reps, self.stage, angle


class ShoulderPressCounter(ExerciseCounter):
    def __init__(self):
        super().__init__("shoulder_press")

    def update(self, landmarks) -> Tuple[int, str, Optional[float]]:
        p = mp.solutions.pose.PoseLandmark
        left = _joint_angle(landmarks, p.LEFT_SHOULDER.value, p.LEFT_ELBOW.value, p.LEFT_WRIST.value)
        right = _joint_angle(landmarks, p.RIGHT_SHOULDER.value, p.RIGHT_ELBOW.value, p.RIGHT_WRIST.value)
        vals = [v for v in [left, right] if v is not None]
        if not vals:
            return self.reps, self.stage, None
        angle = self._smooth(float(np.max(vals)))
        if angle is None:
            return self.reps, self.stage, None
        self._apply_standard_rep_logic(angle <= 100.0, angle >= 160.0)
        return self.reps, self.stage, angle


class LungeCounter(ExerciseCounter):
    def __init__(self):
        super().__init__("lunge")

    def update(self, landmarks) -> Tuple[int, str, Optional[float]]:
        p = mp.solutions.pose.PoseLandmark
        left = _joint_angle(landmarks, p.LEFT_HIP.value, p.LEFT_KNEE.value, p.LEFT_ANKLE.value)
        right = _joint_angle(landmarks, p.RIGHT_HIP.value, p.RIGHT_KNEE.value, p.RIGHT_ANKLE.value)
        vals = [v for v in [left, right] if v is not None]
        if not vals:
            return self.reps, self.stage, None
        l_ankle = _safe_landmark_xy(landmarks, p.LEFT_ANKLE.value)
        r_ankle = _safe_landmark_xy(landmarks, p.RIGHT_ANKLE.value)
        if not (l_ankle and r_ankle):
            return self.reps, self.stage, None
        split_dist = abs(l_ankle[0] - r_ankle[0])
        angle = self._smooth(float(np.min(vals)))
        if angle is None:
            return self.reps, self.stage, None
        self._apply_standard_rep_logic(split_dist >= 0.18 and angle <= 115.0, angle >= 155.0)
        return self.reps, self.stage, angle


class PlankCounter(ExerciseCounter):
    def __init__(self):
        super().__init__("plank")
        self.stage = "HOLD"

    def reset(self) -> None:
        self.reps = 0
        self.stage = "HOLD"
        self._angle_hist = deque(maxlen=SMOOTHING_WINDOW)
        self._rep_armed = False
        self._down_frames = 0

    def update(self, landmarks) -> Tuple[int, str, Optional[float]]:
        p = mp.solutions.pose.PoseLandmark
        left = _joint_angle(landmarks, p.LEFT_SHOULDER.value, p.LEFT_HIP.value, p.LEFT_ANKLE.value)
        right = _joint_angle(landmarks, p.RIGHT_SHOULDER.value, p.RIGHT_HIP.value, p.RIGHT_ANKLE.value)
        vals = [v for v in [left, right] if v is not None]
        if not vals:
            return self.reps, self.stage, None
        angle = self._smooth(float(np.mean(vals)))
        return self.reps, self.stage, angle


class Bird:
    def __init__(self, width: int = 960, height: int = 640,
                 gravity: float = GRAVITY, flap_strength: float = FLAP_STRENGTH):
        self.width = width
        self.height = height
        self.bird_x = int(width * 0.2)
        self.ceiling_y = 30
        self.ground_y = int(height * 0.88)
        self.gravity = gravity
        self.flap_strength = flap_strength
        self.reset()

    def reset(self) -> None:
        self.y_position = float(self.height * 0.45)
        self.velocity = 0.0
        self.game_over = False
        self.game_started = False
        self.countdown_started_at: Optional[float] = None
        self.hover_phase = 0.0

    def start_after_first_rep(self) -> None:
        if not self.game_started:
            self.game_started = True
            if USE_COUNTDOWN:
                self.countdown_started_at = time.time()
            print("[GAME] Started after first rep!")

    def flap(self) -> None:
        if self.game_over:
            return
        if not self.game_started:
            self.start_after_first_rep()
        self.velocity = -self.flap_strength

    def _countdown_remaining(self) -> int:
        if not USE_COUNTDOWN or self.countdown_started_at is None:
            return 0
        elapsed = time.time() - self.countdown_started_at
        remaining = COUNTDOWN_SECONDS - int(elapsed)
        return max(0, remaining)

    def physics_active(self) -> bool:
        if not self.game_started:
            return False
        if not USE_COUNTDOWN:
            return True
        return self._countdown_remaining() == 0

    def update(self) -> None:
        if self.game_over:
            return

        if not self.game_started:
            self.hover_phase += 0.08
            self.y_position = (self.height * 0.45) + (math.sin(self.hover_phase) * 6.0)
            self.velocity = 0.0
            return

        if not self.physics_active():
            self.hover_phase += 0.12
            self.y_position += math.sin(self.hover_phase) * 0.8
            return

        self.velocity += self.gravity
        self.y_position += self.velocity

        # Game over only applies after game has started and physics is active.
        if self.y_position >= self.ground_y or self.y_position <= self.ceiling_y:
            self.game_over = True

    def draw(self, canvas, score: int, exercise_label: str) -> None:
        canvas[:] = (25, 30, 45)
        cv2.line(canvas, (0, self.ceiling_y), (self.width, self.ceiling_y), (120, 120, 255), 2)
        cv2.line(canvas, (0, self.ground_y), (self.width, self.ground_y), (80, 220, 120), 4)

        y_int = int(self.y_position)
        cv2.circle(canvas, (self.bird_x, y_int), 20, (0, 230, 255), -1)
        cv2.circle(canvas, (self.bird_x + 8, y_int - 6), 4, (0, 0, 0), -1)

        cv2.putText(canvas, "BIRD FLIP MODE", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(canvas, f"Score: {score}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, f"Exercise: {exercise_label}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
        cv2.putText(canvas, "Press R to Restart | Q to Quit", (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

        if not self.game_started:
            cv2.putText(canvas, "DO YOUR FIRST REP TO START!", (self.width // 2 - 245, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)
        elif USE_COUNTDOWN and not self.physics_active():
            rem = self._countdown_remaining()
            text = "GO!" if rem == 0 else f"{rem}..."
            cv2.putText(canvas, text, (self.width // 2 - 30, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)

        if self.game_over:
            cv2.putText(canvas, "GAME OVER", (self.width // 2 - 130, self.height // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 80, 255), 3)
            cv2.putText(canvas, "Press R to Restart", (self.width // 2 - 145, self.height // 2 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


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


def _mode_label(mode: str) -> str:
    for _, (m, label) in MODE_CONFIG.items():
        if m == mode:
            return label
    return mode


def draw_hud(frame, mode: str, reps: int, stage: str, angle: Optional[float], fps: float) -> None:
    cv2.rectangle(frame, (10, 10), (430, 170), (0, 0, 0), -1)
    angle_text = "N/A" if angle is None else f"{angle:.1f}"
    cv2.putText(frame, f"Exercise: {_mode_label(mode)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(frame, f"Reps: {reps}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Stage: {stage}", (20, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(frame, f"Angle: {angle_text}", (20, 141),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (280, 141),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    x0, y0 = 450, 10
    cv2.rectangle(frame, (x0, y0), (x0 + 270, y0 + 250), (0, 0, 0), -1)
    cv2.putText(frame, "Press Keys to Switch:", (x0 + 12, y0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    rows = [
        ("1", "Squat", "squat"),
        ("2", "Pushup", "pushup"),
        ("3", "Curl", "curl"),
        ("4", "Shoulder Press", "shoulder_press"),
        ("5", "Lunge", "lunge"),
        ("6", "Plank", "plank"),
    ]
    y = y0 + 58
    for key, label, m in rows:
        active = m == mode
        color = (0, 255, 0) if active else (220, 220, 220)
        marker = ">" if active else " "
        cv2.putText(frame, f"{marker} {key} {label}", (x0 + 14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 30

    cv2.putText(frame, "Bird Flip: ON" if GAME_MODE else "Bird Flip: OFF", (x0 + 12, y0 + 238),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255) if GAME_MODE else (180, 180, 180), 2)


def main() -> None:
    counters = build_counters()
    current_mode = INITIAL_EXERCISE_MODE if INITIAL_EXERCISE_MODE in counters else "squat"
    counters[current_mode].reset()
    prev_reps = 0
    total_score = 0

    bird = Bird()

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
            ok, cam_frame = cap.read()
            if not ok:
                print("Warning: Failed to read frame from webcam.")
                break

            cam_frame = cv2.flip(cam_frame, 1)
            rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    cam_frame,
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
            rep_detected = reps > prev_reps
            prev_reps = reps

            if GAME_MODE and rep_detected:
                total_score += 1
                bird.flap()

            if GAME_MODE:
                bird.update()
                out = np.zeros((bird.height, bird.width, 3), dtype=np.uint8)
                bird.draw(out, score=total_score, exercise_label=_mode_label(current_mode))

                cam_small = cv2.resize(cam_frame, (320, 240))
                out[10:250, bird.width - 330:bird.width - 10] = cam_small
                cv2.rectangle(out, (bird.width - 330, 10), (bird.width - 10, 250), (255, 255, 255), 1)
                cv2.putText(out, "Pose Feed", (bird.width - 320, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                out = cam_frame

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-8)
            prev_time = now

            draw_hud(out, current_mode, reps, stage, angle, fps)
            cv2.imshow("Bird Flip Fitness Game v2", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("r") and GAME_MODE:
                bird.reset()
                total_score = 0
                prev_reps = counters[current_mode].reps
                print("[SYSTEM] Bird Flip game restarted")

            if key in MODE_CONFIG:
                next_mode, next_label = MODE_CONFIG[key]
                if next_mode != current_mode:
                    current_mode = next_mode
                    counters[current_mode].reset()
                    prev_reps = 0
                    print(f"[SYSTEM] Switched to {next_label} Mode")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
