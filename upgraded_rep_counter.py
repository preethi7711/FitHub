"""
Professional-grade squat rep counter prototype using MediaPipe + OpenCV.

Install:
    pip install opencv-python mediapipe numpy

Run:
    python upgraded_rep_counter.py
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
import time

import cv2
import mediapipe as mp
import numpy as np


# ----------------------------- Configuration ----------------------------- #
EXERCISE_MODE = "squat"  # Placeholder for future modes (e.g., "pushup")
CAMERA_INDEX = 0
VISIBILITY_THRESHOLD = 0.5

CALIBRATION_SECONDS = 3.0
SMOOTHING_WINDOW = 5
DOWN_HOLD_FRAMES = 3

# Dynamic thresholds are derived from baseline standing angle.
UP_RATIO_FROM_BASELINE = 0.90
DOWN_RATIO_FROM_BASELINE = 0.72


@dataclass
class RepState:
    reps: int = 0
    stage: str = "UP"
    down_frames: int = 0
    rep_armed: bool = False


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
    lm = landmarks[idx]
    if hasattr(lm, "visibility") and lm.visibility < VISIBILITY_THRESHOLD:
        return None
    return float(lm.x), float(lm.y)


def extract_knee_angle(landmarks) -> Optional[float]:
    """Return a conservative knee angle using the more-bent visible leg."""
    if landmarks is None:
        return None

    pose_lm = mp.solutions.pose.PoseLandmark
    angles: List[float] = []

    l_hip = _safe_landmark_xy(landmarks, pose_lm.LEFT_HIP.value)
    l_knee = _safe_landmark_xy(landmarks, pose_lm.LEFT_KNEE.value)
    l_ankle = _safe_landmark_xy(landmarks, pose_lm.LEFT_ANKLE.value)
    if l_hip and l_knee and l_ankle:
        angles.append(calculate_angle(l_hip, l_knee, l_ankle))

    r_hip = _safe_landmark_xy(landmarks, pose_lm.RIGHT_HIP.value)
    r_knee = _safe_landmark_xy(landmarks, pose_lm.RIGHT_KNEE.value)
    r_ankle = _safe_landmark_xy(landmarks, pose_lm.RIGHT_ANKLE.value)
    if r_hip and r_knee and r_ankle:
        angles.append(calculate_angle(r_hip, r_knee, r_ankle))

    if not angles:
        return None

    return float(np.min(angles))


class RepCounter:
    def __init__(self,
                 smoothing_window: int = SMOOTHING_WINDOW,
                 down_hold_frames: int = DOWN_HOLD_FRAMES,
                 calibration_seconds: float = CALIBRATION_SECONDS):
        self.state = RepState()
        self.smoothing_window = smoothing_window
        self.down_hold_frames = down_hold_frames
        self.calibration_seconds = calibration_seconds

        self.angle_history: Deque[float] = deque(maxlen=self.smoothing_window)
        self.calibration_angles: List[float] = []
        self.calibration_start = time.time()

        self.baseline_angle: Optional[float] = None
        self.up_threshold: Optional[float] = None
        self.down_threshold: Optional[float] = None

    @property
    def is_calibrated(self) -> bool:
        return self.baseline_angle is not None

    def smooth_angle(self, raw_angle: Optional[float]) -> Optional[float]:
        if raw_angle is None:
            return None
        self.angle_history.append(raw_angle)
        return float(np.mean(self.angle_history))

    def _finalize_calibration(self) -> None:
        if self.calibration_angles:
            baseline = float(np.median(self.calibration_angles))
        else:
            baseline = 170.0  # fallback if no stable landmarks during calibration

        baseline = np.clip(baseline, 130.0, 185.0)
        self.baseline_angle = float(baseline)
        self.up_threshold = float(self.baseline_angle * UP_RATIO_FROM_BASELINE)
        self.down_threshold = float(self.baseline_angle * DOWN_RATIO_FROM_BASELINE)

        print(
            f"Calibration complete | baseline={self.baseline_angle:.1f} "
            f"up={self.up_threshold:.1f} down={self.down_threshold:.1f}"
        )

    def update(self, smoothed_angle: Optional[float]) -> None:
        if smoothed_angle is None:
            return

        if not self.is_calibrated:
            elapsed = time.time() - self.calibration_start
            self.calibration_angles.append(smoothed_angle)
            if elapsed >= self.calibration_seconds:
                self._finalize_calibration()
            return

        assert self.up_threshold is not None
        assert self.down_threshold is not None

        # DOWN validation: must hold for N consecutive frames.
        if smoothed_angle <= self.down_threshold:
            self.state.down_frames += 1
            if self.state.down_frames == self.down_hold_frames and self.state.stage != "DOWN":
                self.state.stage = "DOWN"
                self.state.rep_armed = True
                print("DOWN detected")
        else:
            if self.state.stage != "DOWN":
                self.state.down_frames = 0

        # UP transition after validated DOWN counts one rep.
        if self.state.rep_armed and smoothed_angle >= self.up_threshold:
            self.state.stage = "UP"
            self.state.reps += 1
            self.state.rep_armed = False
            self.state.down_frames = 0
            print("UP detected -> Rep Counted")

    def status_text(self) -> str:
        return "Tracking..." if self.is_calibrated else "Calibrating..."


def draw_overlay(frame,
                 counter: RepCounter,
                 smoothed_angle: Optional[float],
                 fps: float) -> None:
    cv2.rectangle(frame, (10, 10), (380, 210), (0, 0, 0), -1)

    angle_text = "N/A" if smoothed_angle is None else f"{smoothed_angle:.1f}"
    cv2.putText(frame, f"Exercise: {EXERCISE_MODE}", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Reps: {counter.state.reps}", (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, f"Stage: {counter.state.stage}", (20, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Knee Angle (smoothed): {angle_text}", (20, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 158),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 200, 0), 2)

    status = counter.status_text()
    status_color = (0, 165, 255) if not counter.is_calibrated else (0, 255, 0)
    cv2.putText(frame, f"Status: {status}", (20, 188),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2)

    if not counter.is_calibrated:
        cv2.putText(frame, "Stand straight for 3 seconds to calibrate",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    else:
        assert counter.up_threshold is not None
        assert counter.down_threshold is not None
        cv2.putText(frame,
                    f"UP>{counter.up_threshold:.0f} DOWN<{counter.down_threshold:.0f}",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)


def main() -> None:
    if EXERCISE_MODE != "squat":
        raise NotImplementedError(f"Unsupported EXERCISE_MODE: {EXERCISE_MODE}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Check device index/permissions.")

    counter = RepCounter()
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
            raw_angle = extract_knee_angle(landmarks)
            smoothed_angle = counter.smooth_angle(raw_angle)
            counter.update(smoothed_angle)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-8)
            prev_time = now

            draw_overlay(frame, counter, smoothed_angle, fps)
            cv2.imshow("Professional AI Rep Counter", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
