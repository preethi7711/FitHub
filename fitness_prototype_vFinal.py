"""
Fitness Prototype vFinal

Install:
    pip install opencv-python mediapipe numpy

Run:
    python fitness_prototype_vFinal.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple
import json
import os
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
WORKOUT_HISTORY_FILE = "workout_history.json"
FEEDBACK_COOLDOWN_SECONDS = 2.0

# XP rules
XP_PER_REP = 10.0
XP_PER_PLANK_SECOND = 2.0


@dataclass
class CounterUpdate:
    reps: int
    stage: str
    metric: Optional[float]  # angle for rep exercises, seconds for plank
    new_reps: int = 0
    plank_delta_seconds: float = 0.0
    feedback: Optional[str] = None


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


def safe_landmark_xy(landmarks, idx: int) -> Optional[Tuple[float, float]]:
    if landmarks is None:
        return None
    lm = landmarks[idx]
    if hasattr(lm, "visibility") and lm.visibility < VISIBILITY_THRESHOLD:
        return None
    return float(lm.x), float(lm.y)


def joint_angle(landmarks, a_idx: int, b_idx: int, c_idx: int) -> Optional[float]:
    a = safe_landmark_xy(landmarks, a_idx)
    b = safe_landmark_xy(landmarks, b_idx)
    c = safe_landmark_xy(landmarks, c_idx)
    if a and b and c:
        return calculate_angle(a, b, c)
    return None


class ExerciseCounter(ABC):
    def __init__(self, name: str):
        self.name = name
        self.reps = 0
        self.stage = "UP"
        self._rep_armed = False
        self._down_frames = 0
        self._angle_hist = []
        self._last_feedback_ts = 0.0

    def _smooth(self, angle: Optional[float]) -> Optional[float]:
        if angle is None:
            return None
        self._angle_hist.append(float(angle))
        if len(self._angle_hist) > SMOOTHING_WINDOW:
            self._angle_hist.pop(0)
        return float(np.mean(self._angle_hist))

    def _try_feedback(self, text: str) -> Optional[str]:
        now = time.time()
        if now - self._last_feedback_ts >= FEEDBACK_COOLDOWN_SECONDS:
            self._last_feedback_ts = now
            return text
        return None

    @abstractmethod
    def update(self, landmarks, dt: float) -> CounterUpdate:
        raise NotImplementedError


class SquatCounter(ExerciseCounter):
    def __init__(self):
        super().__init__("squat")
        self._rep_min_angle = 180.0

    def update(self, landmarks, dt: float) -> CounterUpdate:
        p = mp.solutions.pose.PoseLandmark
        l = joint_angle(landmarks, p.LEFT_HIP.value, p.LEFT_KNEE.value, p.LEFT_ANKLE.value)
        r = joint_angle(landmarks, p.RIGHT_HIP.value, p.RIGHT_KNEE.value, p.RIGHT_ANKLE.value)
        vals = [v for v in [l, r] if v is not None]
        if not vals:
            return CounterUpdate(self.reps, self.stage, None)

        angle = self._smooth(float(np.min(vals)))
        if angle is None:
            return CounterUpdate(self.reps, self.stage, None)

        feedback = None
        new_reps = 0

        if angle <= 120.0:
            self._down_frames += 1
            self._rep_min_angle = min(self._rep_min_angle, angle)
            if self._down_frames >= DOWN_HOLD_FRAMES and self.stage != "DOWN":
                self.stage = "DOWN"
                self._rep_armed = True
        else:
            if self.stage != "DOWN":
                self._down_frames = 0
                self._rep_min_angle = 180.0

        if self._rep_armed and angle >= 155.0:
            self.reps += 1
            new_reps = 1
            self.stage = "UP"
            self._rep_armed = False
            self._down_frames = 0
            if self._rep_min_angle > 100.0:
                feedback = self._try_feedback("Go deeper!")
            self._rep_min_angle = 180.0

        return CounterUpdate(self.reps, self.stage, angle, new_reps=new_reps, feedback=feedback)


class PushupCounter(ExerciseCounter):
    def __init__(self):
        super().__init__("pushup")
        self._rep_min_angle = 180.0

    def update(self, landmarks, dt: float) -> CounterUpdate:
        p = mp.solutions.pose.PoseLandmark
        l = joint_angle(landmarks, p.LEFT_SHOULDER.value, p.LEFT_ELBOW.value, p.LEFT_WRIST.value)
        r = joint_angle(landmarks, p.RIGHT_SHOULDER.value, p.RIGHT_ELBOW.value, p.RIGHT_WRIST.value)
        vals = [v for v in [l, r] if v is not None]
        if not vals:
            return CounterUpdate(self.reps, self.stage, None)

        angle = self._smooth(float(np.min(vals)))
        if angle is None:
            return CounterUpdate(self.reps, self.stage, None)

        feedback = None
        new_reps = 0

        if angle <= 120.0:
            self._down_frames += 1
            self._rep_min_angle = min(self._rep_min_angle, angle)
            if self._down_frames >= DOWN_HOLD_FRAMES and self.stage != "DOWN":
                self.stage = "DOWN"
                self._rep_armed = True
        else:
            if self.stage != "DOWN":
                self._down_frames = 0
                self._rep_min_angle = 180.0

        if self._rep_armed and angle >= 160.0:
            self.reps += 1
            new_reps = 1
            self.stage = "UP"
            self._rep_armed = False
            self._down_frames = 0
            if self._rep_min_angle > 100.0:
                feedback = self._try_feedback("Lower your chest!")
            self._rep_min_angle = 180.0

        return CounterUpdate(self.reps, self.stage, angle, new_reps=new_reps, feedback=feedback)


class CurlCounter(ExerciseCounter):
    def __init__(self, arm: str = CURL_ARM):
        super().__init__("curl")
        self.arm = arm.lower()

    def update(self, landmarks, dt: float) -> CounterUpdate:
        p = mp.solutions.pose.PoseLandmark
        if self.arm == "left":
            raw = joint_angle(landmarks, p.LEFT_SHOULDER.value, p.LEFT_ELBOW.value, p.LEFT_WRIST.value)
        else:
            raw = joint_angle(landmarks, p.RIGHT_SHOULDER.value, p.RIGHT_ELBOW.value, p.RIGHT_WRIST.value)

        angle = self._smooth(raw)
        if angle is None:
            return CounterUpdate(self.reps, self.stage, None)

        new_reps = 0
        if angle <= 60.0:
            self._down_frames += 1
            if self._down_frames >= DOWN_HOLD_FRAMES and self.stage != "DOWN":
                self.stage = "DOWN"
                self._rep_armed = True
        else:
            if self.stage != "DOWN":
                self._down_frames = 0

        if self._rep_armed and angle >= 145.0:
            self.reps += 1
            new_reps = 1
            self.stage = "UP"
            self._rep_armed = False
            self._down_frames = 0

        return CounterUpdate(self.reps, self.stage, angle, new_reps=new_reps)


class ShoulderPressCounter(ExerciseCounter):
    def __init__(self):
        super().__init__("shoulder_press")
        self._rep_max_angle = 0.0

    def update(self, landmarks, dt: float) -> CounterUpdate:
        p = mp.solutions.pose.PoseLandmark
        l = joint_angle(landmarks, p.LEFT_SHOULDER.value, p.LEFT_ELBOW.value, p.LEFT_WRIST.value)
        r = joint_angle(landmarks, p.RIGHT_SHOULDER.value, p.RIGHT_ELBOW.value, p.RIGHT_WRIST.value)
        vals = [v for v in [l, r] if v is not None]
        if not vals:
            return CounterUpdate(self.reps, self.stage, None)

        angle = self._smooth(float(np.max(vals)))
        if angle is None:
            return CounterUpdate(self.reps, self.stage, None)

        feedback = None
        new_reps = 0

        if angle <= 110.0:
            self._down_frames += 1
            if self._down_frames >= DOWN_HOLD_FRAMES and self.stage != "DOWN":
                self.stage = "DOWN"
                self._rep_armed = True
                self._rep_max_angle = angle
        else:
            if self.stage != "DOWN":
                self._down_frames = 0

        if self._rep_armed:
            self._rep_max_angle = max(self._rep_max_angle, angle)

        if self._rep_armed and angle >= 155.0:
            self.reps += 1
            new_reps = 1
            self.stage = "UP"
            self._rep_armed = False
            self._down_frames = 0
            if self._rep_max_angle < 165.0:
                feedback = self._try_feedback("Press higher!")
            self._rep_max_angle = 0.0

        return CounterUpdate(self.reps, self.stage, angle, new_reps=new_reps, feedback=feedback)


class LungeCounter(ExerciseCounter):
    def __init__(self):
        super().__init__("lunge")

    def update(self, landmarks, dt: float) -> CounterUpdate:
        p = mp.solutions.pose.PoseLandmark
        l = joint_angle(landmarks, p.LEFT_HIP.value, p.LEFT_KNEE.value, p.LEFT_ANKLE.value)
        r = joint_angle(landmarks, p.RIGHT_HIP.value, p.RIGHT_KNEE.value, p.RIGHT_ANKLE.value)
        vals = [v for v in [l, r] if v is not None]
        if not vals:
            return CounterUpdate(self.reps, self.stage, None)

        l_ankle = safe_landmark_xy(landmarks, p.LEFT_ANKLE.value)
        r_ankle = safe_landmark_xy(landmarks, p.RIGHT_ANKLE.value)
        if not (l_ankle and r_ankle):
            return CounterUpdate(self.reps, self.stage, None)

        split_dist = abs(l_ankle[0] - r_ankle[0])
        angle = self._smooth(float(np.min(vals)))
        if angle is None:
            return CounterUpdate(self.reps, self.stage, None)

        new_reps = 0
        if split_dist >= 0.18 and angle <= 115.0:
            self._down_frames += 1
            if self._down_frames >= DOWN_HOLD_FRAMES and self.stage != "DOWN":
                self.stage = "DOWN"
                self._rep_armed = True
        else:
            if self.stage != "DOWN":
                self._down_frames = 0

        if self._rep_armed and angle >= 155.0:
            self.reps += 1
            new_reps = 1
            self.stage = "UP"
            self._rep_armed = False
            self._down_frames = 0

        return CounterUpdate(self.reps, self.stage, angle, new_reps=new_reps)


class PlankCounter(ExerciseCounter):
    def __init__(self):
        super().__init__("plank")
        self.stage = "BREAK"
        self.plank_seconds = 0.0

    def update(self, landmarks, dt: float) -> CounterUpdate:
        p = mp.solutions.pose.PoseLandmark
        l = joint_angle(landmarks, p.LEFT_SHOULDER.value, p.LEFT_HIP.value, p.LEFT_ANKLE.value)
        r = joint_angle(landmarks, p.RIGHT_SHOULDER.value, p.RIGHT_HIP.value, p.RIGHT_ANKLE.value)
        vals = [v for v in [l, r] if v is not None]
        if not vals:
            self.stage = "BREAK"
            return CounterUpdate(self.reps, self.stage, None, plank_delta_seconds=0.0)

        body_angle = self._smooth(float(np.mean(vals)))
        if body_angle is None:
            self.stage = "BREAK"
            return CounterUpdate(self.reps, self.stage, None, plank_delta_seconds=0.0)

        feedback = None
        delta = 0.0
        if body_angle > 160.0:
            self.stage = "HOLD"
            self.plank_seconds += dt
            delta = dt
        else:
            self.stage = "BREAK"
            feedback = self._try_feedback("Fix posture!")

        return CounterUpdate(
            reps=0,
            stage=self.stage,
            metric=self.plank_seconds,
            plank_delta_seconds=delta,
            feedback=feedback,
        )


def build_counters() -> Dict[str, ExerciseCounter]:
    return {
        "squat": SquatCounter(),
        "pushup": PushupCounter(),
        "curl": CurlCounter(),
        "shoulder_press": ShoulderPressCounter(),
        "lunge": LungeCounter(),
        "plank": PlankCounter(),
    }


def mode_label(mode: str) -> str:
    labels = {
        "squat": "Squat",
        "pushup": "Pushup",
        "curl": "Curl",
        "shoulder_press": "Shoulder Press",
        "lunge": "Lunge",
        "plank": "Plank",
    }
    return labels.get(mode, mode)


def save_session(stats: Dict[str, float], total_xp: float) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stats": stats,
        "total_xp": int(total_xp),
    }

    history = []
    if os.path.exists(WORKOUT_HISTORY_FILE):
        try:
            with open(WORKOUT_HISTORY_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    history = loaded
                elif isinstance(loaded, dict):
                    history = [loaded]
        except Exception:
            history = []

    history.append(payload)
    with open(WORKOUT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def draw_mode_menu(frame, current_mode: str) -> None:
    x0, y0 = 430, 10
    cv2.rectangle(frame, (x0, y0), (x0 + 290, y0 + 250), (0, 0, 0), -1)
    cv2.putText(frame, "Modes:", (x0 + 12, y0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    rows = [
        ("1", "Squat", "squat"),
        ("2", "Pushup", "pushup"),
        ("3", "Curl", "curl"),
        ("4", "Shoulder Press", "shoulder_press"),
        ("5", "Lunge", "lunge"),
        ("6", "Plank", "plank"),
    ]
    y = y0 + 58
    for key, label, mode in rows:
        active = mode == current_mode
        color = (0, 255, 0) if active else (220, 220, 220)
        marker = ">" if active else " "
        cv2.putText(frame, f"{marker} {key} {label}", (x0 + 14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 30

    cv2.putText(frame, "E End Workout | Q Quit", (x0 + 12, y0 + 238),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 2)


def draw_overlay(frame,
                 mode: str,
                 result: CounterUpdate,
                 total_xp: float,
                 feedback: Optional[str],
                 fps: float) -> None:
    cv2.rectangle(frame, (10, 10), (410, 180), (0, 0, 0), -1)

    cv2.putText(frame, f"Exercise: {mode_label(mode)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    if mode == "plank":
        plank_time = 0.0 if result.metric is None else float(result.metric)
        cv2.putText(frame, f"Plank Time: {plank_time:.1f} sec", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"Reps: {result.reps}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    angle_text = "N/A" if result.metric is None else f"{result.metric:.1f}"
    metric_label = "Body Angle" if mode == "plank" else "Angle"
    cv2.putText(frame, f"Stage: {result.stage}", (20, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(frame, f"{metric_label}: {angle_text}", (20, 141),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2)
    cv2.putText(frame, f"Total XP: {int(total_xp)}", (20, 172),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 200, 0), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (300, 172),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 2)

    if feedback:
        cv2.putText(frame, feedback, (20, frame.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 3)

    draw_mode_menu(frame, mode)


def draw_summary_overlay(frame, stats: Dict[str, float], total_xp: float) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (40, 40), (w - 40, h - 40), (0, 0, 0), -1)
    cv2.putText(frame, "WORKOUT COMPLETE!", (70, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

    lines = [
        f"Squats: {int(stats['squat'])}",
        f"Pushups: {int(stats['pushup'])}",
        f"Curls: {int(stats['curl'])}",
        f"Shoulder Press: {int(stats['shoulder_press'])}",
        f"Lunges: {int(stats['lunge'])}",
        f"Plank Hold: {int(stats['plank_seconds'])} sec",
        f"Total XP: {int(total_xp)}",
    ]
    y = 145
    for line in lines:
        cv2.putText(frame, line, (70, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        y += 45

    cv2.putText(frame, "Press Q to quit", (70, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)


def main() -> None:
    counters = build_counters()
    current_mode = INITIAL_EXERCISE_MODE if INITIAL_EXERCISE_MODE in counters else "squat"

    session_stats: Dict[str, float] = {
        "squat": 0,
        "pushup": 0,
        "curl": 0,
        "shoulder_press": 0,
        "lunge": 0,
        "plank_seconds": 0.0,
    }
    total_xp = 0.0

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Check device index/permissions.")

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    prev_time = time.time()
    last_feedback = ""
    feedback_until = 0.0
    workout_ended = False
    session_saved = False

    last_result = CounterUpdate(reps=0, stage="UP", metric=None)

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
            now = time.time()
            dt = max(now - prev_time, 1e-3)
            fps = 1.0 / dt
            prev_time = now

            if not workout_ended:
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
                counter = counters[current_mode]
                last_result = counter.update(landmarks, dt)

                if current_mode == "plank":
                    session_stats["plank_seconds"] += last_result.plank_delta_seconds
                    total_xp += last_result.plank_delta_seconds * XP_PER_PLANK_SECOND
                else:
                    if last_result.new_reps > 0:
                        session_stats[current_mode] += last_result.new_reps
                        total_xp += last_result.new_reps * XP_PER_REP

                if last_result.feedback:
                    last_feedback = last_result.feedback
                    feedback_until = now + 1.3

                active_feedback = last_feedback if now < feedback_until else None
                draw_overlay(frame, current_mode, last_result, total_xp, active_feedback, fps)
            else:
                draw_summary_overlay(frame, session_stats, total_xp)

            cv2.imshow("Fitness Prototype vFinal", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("e") and not workout_ended:
                workout_ended = True
                if not session_saved:
                    save_session(session_stats, total_xp)
                    session_saved = True
                    print("[SYSTEM] Workout ended and saved to workout_history.json")

            if not workout_ended:
                mode_map = {
                    ord("1"): "squat",
                    ord("2"): "pushup",
                    ord("3"): "curl",
                    ord("4"): "shoulder_press",
                    ord("5"): "lunge",
                    ord("6"): "plank",
                }
                if key in mode_map:
                    current_mode = mode_map[key]
                    print(f"[SYSTEM] Switched to {mode_label(current_mode)} Mode")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
