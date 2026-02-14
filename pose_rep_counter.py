# pose_rep_counter.py
"""
AI Rep Counter (Squats) using MediaPipe + OpenCV

Install:
    pip install opencv-python mediapipe numpy

Run:
    python pose_rep_counter.py
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import mediapipe as mp
import numpy as np


# ----------------------------- Configuration ----------------------------- #
UP_THRESHOLD = 155.0
DOWN_THRESHOLD = 125.0
VISIBILITY_THRESHOLD = 0.5
CAMERA_INDEX = 0


# ------------------------------ Data Model ------------------------------- #
@dataclass
class RepCounterState:
    reps: int = 0
    stage: str = "UP"  # "UP" | "DOWN"
    knee_angle: Optional[float] = None


COUNTER_STATE = RepCounterState()


# ------------------------------ Core Logic ------------------------------- #
def calculate_angle(a, b, c) -> float:
    """
    Calculate angle ABC (in degrees) from 2D points a, b, c.
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosine = float(np.dot(ba, bc) / denom)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _safe_landmark_xy(landmarks, idx: int) -> Optional[Tuple[float, float]]:
    """
    Return normalized (x, y) for a landmark if visibility is reliable.
    """
    lm = landmarks[idx]
    if hasattr(lm, "visibility") and lm.visibility < VISIBILITY_THRESHOLD:
        return None
    return float(lm.x), float(lm.y)


def detect_squat_reps(landmarks) -> Tuple[int, str, Optional[float]]:
    """
    Detect squat reps using a state machine:
    - angle > 160 => UP
    - angle < 90  => DOWN
    - rep += 1 only on transition DOWN -> UP
    """
    global COUNTER_STATE

    # Safety check: missing landmarks
    if landmarks is None:
        return COUNTER_STATE.reps, COUNTER_STATE.stage, COUNTER_STATE.knee_angle

    pose_lm = mp.solutions.pose.PoseLandmark

    # Collect left and right knee angles when possible
    angles: List[float] = []

    # Left leg
    l_hip = _safe_landmark_xy(landmarks, pose_lm.LEFT_HIP.value)
    l_knee = _safe_landmark_xy(landmarks, pose_lm.LEFT_KNEE.value)
    l_ankle = _safe_landmark_xy(landmarks, pose_lm.LEFT_ANKLE.value)
    if l_hip and l_knee and l_ankle:
        angles.append(calculate_angle(l_hip, l_knee, l_ankle))

    # Right leg
    r_hip = _safe_landmark_xy(landmarks, pose_lm.RIGHT_HIP.value)
    r_knee = _safe_landmark_xy(landmarks, pose_lm.RIGHT_KNEE.value)
    r_ankle = _safe_landmark_xy(landmarks, pose_lm.RIGHT_ANKLE.value)
    if r_hip and r_knee and r_ankle:
        angles.append(calculate_angle(r_hip, r_knee, r_ankle))

    # Safety check: no valid knee angle this frame
    if not angles:
        return COUNTER_STATE.reps, COUNTER_STATE.stage, COUNTER_STATE.knee_angle

    # Use the lower of the visible knee angles; averaging can hide a true squat
    # when one leg is partially occluded or noisier than the other.
    knee_angle = float(np.min(angles))
    COUNTER_STATE.knee_angle = knee_angle

    # State machine (prevents false multiple counts)
    if knee_angle < DOWN_THRESHOLD:
        COUNTER_STATE.stage = "DOWN"
    elif knee_angle > UP_THRESHOLD:
        if COUNTER_STATE.stage == "DOWN":
            COUNTER_STATE.reps += 1
        COUNTER_STATE.stage = "UP"

    return COUNTER_STATE.reps, COUNTER_STATE.stage, COUNTER_STATE.knee_angle


# --------------------------------- App ---------------------------------- #
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Check device index/permissions.")

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

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

            # Draw skeleton overlay
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

            reps, stage, angle = detect_squat_reps(
                results.pose_landmarks.landmark if results.pose_landmarks else None
            )

            # Top-left UI box
            cv2.rectangle(frame, (10, 10), (300, 135), (0, 0, 0), -1)
            cv2.putText(frame, f"Reps: {reps}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Stage: {stage}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            angle_text = f"Knee Angle: {angle:.1f}" if angle is not None else "Knee Angle: N/A"
            cv2.putText(frame, angle_text, (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, "Press 'q' to quit",
                        (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

            cv2.imshow("AI Rep Counter (Squats)", frame)

            # Quit safely
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
