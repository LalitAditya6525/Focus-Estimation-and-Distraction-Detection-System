import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time
import os
import threading
from collections import deque
from datetime import datetime

# Notification support
try:
    from plyer import notification as plyer_notify
    NOTIFY_AVAILABLE = True
except ImportError:
    NOTIFY_AVAILABLE = False
    print("[Warning] 'plyer' not installed. Run: pip install plyer")
    print("[Info] Falling back to on-screen alerts only.\n")

# -------------------------------
# NOTIFICATION HELPER
# -------------------------------

# Cooldown tracker: { "key": last_sent_timestamp }
_notify_cooldowns = {}

def send_notification(title, message, timeout=6, cooldown_key=None, cooldown_sec=30):
    """
    Send a desktop notification. Respects cooldowns to avoid spamming.
    Runs in a background thread so it never blocks the video loop.
    """
    now = time.time()
    if cooldown_key:
        last = _notify_cooldowns.get(cooldown_key, 0)
        if now - last < cooldown_sec:
            return  # Still in cooldown, skip
        _notify_cooldowns[cooldown_key] = now

    if not NOTIFY_AVAILABLE:
        return

    def _fire():
        try:
            plyer_notify.notify(
                title=title,
                message=message,
                app_name="Focus Monitor",
                timeout=timeout,
            )
        except Exception as e:
            pass  # Silently fail if notification errors out

    threading.Thread(target=_fire, daemon=True).start()


# Human-friendly notification messages (rotated to avoid monotony)
DISTRACTION_MESSAGES = [
    "Looks like your mind wandered — no worries! Let's refocus. 🎯",
    "Hey there! The screen misses you. Come back when you're ready. 👀",
    "Quick check-in: you seem a bit distracted. You've got this! 💪",
    "Drifted off? Take a breath and come back — you're doing great. 🌟",
    "Attention check! Refocus and keep going strong. 🔔",
]

BLINK_MESSAGES = [
    "Your eyes look tired 😴 — remember to blink and take a short break!",
    "Eyes feeling heavy? Look away for 20 seconds to recharge. 👁️",
    "Eye fatigue detected. Try the 20-20-20 rule: look 20ft away for 20 sec!",
]

MILESTONE_MESSAGES = {
    5:  "You've been distracted 5 times. Take a 5-minute break — you deserve it! ☕",
    10: "10 distractions logged. Maybe a short walk would help clear your head? 🚶",
    20: "You seem to be struggling to focus today. That's okay — rest is productive too! 🛌",
}

POSITIVE_MESSAGES = [
    "Amazing! You've been focused for 5 minutes straight. Keep it up! 🔥",
    "Great work — 5 solid minutes of focus! You're in the zone. ⚡",
    "5 minutes of pure focus! Your future self thanks you. 🏆",
]

_distraction_msg_idx = 0
_blink_msg_idx = 0
_positive_msg_idx = 0

def get_rotating(messages, idx_name):
    """Return messages in rotation."""
    global _distraction_msg_idx, _blink_msg_idx, _positive_msg_idx
    idx_map = {
        "distraction": "_distraction_msg_idx",
        "blink": "_blink_msg_idx",
        "positive": "_positive_msg_idx",
    }
    if idx_name == "distraction":
        msg = messages[_distraction_msg_idx % len(messages)]
        _distraction_msg_idx += 1
    elif idx_name == "blink":
        msg = messages[_blink_msg_idx % len(messages)]
        _blink_msg_idx += 1
    else:
        msg = messages[_positive_msg_idx % len(messages)]
        _positive_msg_idx += 1
    return msg


# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("Models/Test2/svm.pkl")
scaler = joblib.load("Models/Test2/scaler.pkl")

# -------------------------------
# OUTPUT FILE
# -------------------------------
output_file = "Results/realtime_results_final.csv"

if not os.path.exists(output_file):
    pd.DataFrame(columns=[
        "EAR", "Blink Rate", "Gaze Direction", "Head Angle",
        "Posture", "Emotion", "Screen Time",
        "Distraction Count", "Focus Level", "Prediction"
    ]).to_csv(output_file, index=False)

# -------------------------------
# BUFFER FOR SMOOTHING
# -------------------------------
score_buffer = deque(maxlen=5)

# -------------------------------
# MEDIAPIPE SETUP
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = 468
RIGHT_IRIS = 473

# -------------------------------
# FUNCTIONS
# -------------------------------
def get_face_bbox(landmarks, w, h):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    return x1, y1, x2, y2

def calculate_EAR(landmarks, eye, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C) if C != 0 else 0

# -------------------------------
# CAMERA SETUP
# -------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

session_start_time = time.time()
session_start_datetime = datetime.now()

distraction_count = 0
blink_count = 0
frame_count = 0
blink_frames = 0
focused_frames = 0
distracted_frames = 0

BLINK_THRESHOLD = 0.2
DISTRACTION_THRESHOLD_SECONDS = 2.0

distraction_start_time = None
distraction_counted = False

# Positive focus streak tracking
focus_streak_start = None
_positive_notified_at = 0  # timestamp of last positive notification

# Milestone tracking
_milestones_notified = set()

print("Starting Real-Time Focus Detection System...")
print(f"Desktop notifications: {'Enabled ✓' if NOTIFY_AVAILABLE else 'Disabled (install plyer)'}")
print("Press 'q' or ESC to exit\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        gaze = "Away"
        posture = "Straight"
        emotion = "Neutral"
        head_angle = 0.0
        EAR = 0.0
        blink_rate = 0.0
        show_details = True
        warning_text = ""

        # -------------------------------
        # FACE ANALYSIS
        # -------------------------------
        if results.multi_face_landmarks:
            if len(results.multi_face_landmarks) > 1:
                warning_text = "MULTIPLE FACES DETECTED!"
                show_details = False
                send_notification(
                    "Multiple People Detected 👥",
                    "It looks like someone else is in the frame. Please make sure only you are visible!",
                    cooldown_key="multi_face", cooldown_sec=60
                )
            else:
                landmarks = results.multi_face_landmarks[0].landmark

                x1, y1, x2, y2 = get_face_bbox(landmarks, w, h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # EAR
                leftEAR = calculate_EAR(landmarks, LEFT_EYE, w, h)
                rightEAR = calculate_EAR(landmarks, RIGHT_EYE, w, h)
                EAR = (leftEAR + rightEAR) / 2.0

                # Blink
                if EAR < BLINK_THRESHOLD:
                    blink_frames += 1
                else:
                    if blink_frames >= 2:
                        blink_count += 1
                    blink_frames = 0

                blink_rate = blink_count * 60 / (time.time() - session_start_time + 1)

                # Eye fatigue notification (very low EAR sustained)
                if EAR < 0.18:
                    send_notification(
                        "Eye Fatigue Detected 👁️",
                        get_rotating(BLINK_MESSAGES, "blink"),
                        cooldown_key="eye_fatigue", cooldown_sec=45
                    )

                # Head & Posture
                nose = landmarks[1]
                head_angle = (nose.x - 0.5) * 60
                posture = "Straight" if abs(head_angle) < 12 else "Slouch"

                if posture == "Slouch":
                    send_notification(
                        "Posture Check 🧍",
                        "You're tilting your head a bit — sit up straight for better focus and comfort!",
                        cooldown_key="posture", cooldown_sec=60
                    )

                # Gaze
                left_iris = landmarks[LEFT_IRIS]
                right_iris = landmarks[RIGHT_IRIS]
                avg_x = (left_iris.x + right_iris.x) / 2
                avg_y = (left_iris.y + right_iris.y) / 2

                gaze = "Screen" if (0.38 < avg_x < 0.62 and 0.35 < avg_y < 0.65) else "Away"

                # Draw landmarks
                for eye in [LEFT_EYE, RIGHT_EYE]:
                    for idx in eye:
                        cv2.circle(frame, (int(landmarks[idx].x * w), int(landmarks[idx].y * h)), 2, (0, 255, 0), -1)
                cv2.circle(frame, (int(left_iris.x * w), int(left_iris.y * h)), 6, (0, 0, 255), -1)
                cv2.circle(frame, (int(right_iris.x * w), int(right_iris.y * h)), 6, (0, 0, 255), -1)

                emotion = np.random.choice(["Happy", "Neutral", "Sad", "Focused"])
        else:
            warning_text = "NO FACE DETECTED!"
            show_details = False
            send_notification(
                "Where'd you go? 🙈",
                "We can't see your face! Make sure you're in front of the camera to track your focus.",
                cooldown_key="no_face", cooldown_sec=30
            )

        # -------------------------------
        # DISTRACTION COUNTING
        # -------------------------------
        current_time = time.time()

        if gaze == "Away":
            if distraction_start_time is None:
                distraction_start_time = current_time
                distraction_counted = False

            if (current_time - distraction_start_time >= DISTRACTION_THRESHOLD_SECONDS) and not distraction_counted:
                distraction_count += 1
                distraction_counted = True

                # Distraction notification
                send_notification(
                    "Hey, You Drifted! 🔔",
                    get_rotating(DISTRACTION_MESSAGES, "distraction"),
                    cooldown_key="distraction", cooldown_sec=30
                )

                # Milestone notifications
                if distraction_count in MILESTONE_MESSAGES and distraction_count not in _milestones_notified:
                    _milestones_notified.add(distraction_count)
                    send_notification(
                        f"Distraction Milestone: {distraction_count}x 📊",
                        MILESTONE_MESSAGES[distraction_count],
                        cooldown_key=f"milestone_{distraction_count}", cooldown_sec=5
                    )

            # Reset focus streak
            focus_streak_start = None
        else:
            distraction_start_time = None
            distraction_counted = False

            # Track positive focus streak
            if focus_streak_start is None:
                focus_streak_start = current_time

            focused_duration = current_time - focus_streak_start
            # Fire positive notification every 5 minutes of continuous focus
            if focused_duration >= 300 and (current_time - _positive_notified_at) >= 300:
                _positive_notified_at = current_time
                send_notification(
                    "You're on Fire! 🔥",
                    get_rotating(POSITIVE_MESSAGES, "positive"),
                    cooldown_key="positive_focus", cooldown_sec=300
                )

        # -------------------------------
        # PREDICTION & DISPLAY
        # -------------------------------
        if show_details:
            Eye_Attention_Score = 1 if gaze == "Screen" else 0
            Face_Orientation_Score = 1 if abs(head_angle) < 12 else 0
            Posture_Score = 1 if posture == "Straight" else 0
            emotion_score_map = {"Happy": 1.0, "Neutral": 0.7, "Sad": 0.3, "Focused": 0.9}
            Emotion_Score = emotion_score_map.get(emotion, 0.5)

            Behavior_Score = (Eye_Attention_Score + Face_Orientation_Score + Posture_Score + Emotion_Score) / 4.0

            features = pd.DataFrame(
                [[Eye_Attention_Score, Face_Orientation_Score, Emotion_Score, Posture_Score, Behavior_Score]],
                columns=["Eye_Attention_Score", "Face_Orientation_Score", "Emotion_Score", "Posture_Score", "Behavior_Score"]
            )

            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]

            current_label = "Distracted" if (gaze == "Away" or EAR < 0.18) else ("Focused" if prediction == 1 else "Distracted")
            color = (0, 255, 0) if current_label == "Focused" else (0, 0, 255)

            if current_label == "Focused":
                focused_frames += 1
            else:
                distracted_frames += 1

            focus_level = Eye_Attention_Score * 40 + Face_Orientation_Score * 25 + Posture_Score * 15 + Emotion_Score * 20
            if gaze == "Away":
                focus_level -= 35
            if EAR < 0.2:
                focus_level -= 25
            focus_level = max(0, min(100, focus_level))

            score_buffer.append(focus_level)
            smooth_focus = int(np.mean(score_buffer))
        else:
            current_label = "Warning"
            color = (0, 0, 255)
            smooth_focus = 0

        # -------------------------------
        # ON-SCREEN DISPLAY
        # -------------------------------
        if not show_details:
            cv2.putText(frame, warning_text, (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
            cv2.putText(frame, "Please position yourself properly", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        else:
            cv2.putText(frame, f"Status: {current_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Focus Level: {smooth_focus}%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, f"EAR: {EAR:.3f} | Blink Rate: {blink_rate:.1f}/min", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Gaze: {gaze} | Head: {head_angle:+.1f}°", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Posture: {posture} | Emotion: {emotion}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Distractions: {distraction_count}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)

            # Progress Bar
            bar_x, bar_y = 20, 230
            bar_w, bar_h = 350, 30
            fill = int(smooth_focus / 100 * bar_w)
            bar_color = (0, 255, 0) if smooth_focus > 70 else (0, 255, 255) if smooth_focus > 40 else (0, 0, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)

            # # On-screen distraction popup (kept as fallback visual alert)
            # if current_label == "Distracted":
            #     overlay = frame.copy()
            #     alpha = 0.65
            #     popup_w, popup_h = 560, 200
            #     popup_x1 = w // 2 - popup_w // 2
            #     popup_y1 = h // 2 - popup_h // 2
            #     popup_x2 = popup_x1 + popup_w
            #     popup_y2 = popup_y1 + popup_h

            #     cv2.rectangle(overlay, (popup_x1, popup_y1), (popup_x2, popup_y2), (0, 0, 255), -1)
            #     frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            #     cv2.rectangle(frame, (popup_x1, popup_y1), (popup_x2, popup_y2), (255, 255, 255), 6)

            #     cv2.putText(frame, "YOU ARE DISTRACTED!",
            #                 (w // 2 - 245, h // 2 - 30),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1.85, (255, 255, 255), 5, cv2.LINE_AA)
            #     cv2.putText(frame, "FOCUS BACK ON THE SCREEN",
            #                 (w // 2 - 235, h // 2 + 45),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 4, cv2.LINE_AA)

        cv2.imshow("Focus Detection System - Real-time Analysis", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

        frame_count += 1
        if frame_count % 10 == 0 and show_details:
            pd.DataFrame([{
                "EAR": round(EAR, 4),
                "Blink Rate": round(blink_rate, 2),
                "Gaze Direction": gaze,
                "Head Angle": round(head_angle, 2),
                "Posture": posture,
                "Emotion": emotion,
                "Screen Time": int(time.time() - session_start_time),
                "Distraction Count": distraction_count,
                "Focus Level": smooth_focus,
                "Prediction": current_label
            }]).to_csv(output_file, mode='a', header=False, index=False)

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Final session summary notification
    session_end_time = time.time()
    total_seconds = session_end_time - session_start_time
    focused_time_min = (focused_frames / max(frame_count, 1)) * (total_seconds / 60)
    distracted_time_min = (distracted_frames / max(frame_count, 1)) * (total_seconds / 60)

    send_notification(
        "Session Complete! 🎉",
        f"You stayed focused for {focused_time_min:.1f} min and had {distraction_count} distractions. Great effort today!",
        cooldown_key="session_end", cooldown_sec=0
    )

    print("\n" + "="*70)
    print("                    SESSION REPORT")
    print("="*70)
    print(f"Session Start       : {session_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Session End         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Duration      : {total_seconds/60:.2f} minutes")
    print("-" * 70)
    print(f"Focused Time        : {focused_time_min:.2f} minutes")
    print(f"Distracted Time     : {distracted_time_min:.2f} minutes")
    print(f"Distraction Count   : {distraction_count}")
    print(f"Average Focus Level : {int(np.mean(list(score_buffer))) if score_buffer else 0}%")
    print("="*70)
    print("Session data saved to:", output_file)