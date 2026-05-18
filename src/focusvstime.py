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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Notification support
try:
    from plyer import notification as plyer_notify
    NOTIFY_AVAILABLE = True
except ImportError:
    NOTIFY_AVAILABLE = False
    print("[Warning] 'plyer' not installed. Run: pip install plyer")

# -------------------------------
# NOTIFICATION HELPER (unchanged)
# -------------------------------
_notify_cooldowns = {}
def send_notification(title, message, timeout=6, cooldown_key=None, cooldown_sec=30):
    now = time.time()
    if cooldown_key:
        last = _notify_cooldowns.get(cooldown_key, 0)
        if now - last < cooldown_sec:
            return
        _notify_cooldowns[cooldown_key] = now
    if not NOTIFY_AVAILABLE:
        return
    def _fire():
        try:
            plyer_notify.notify(title=title, message=message, app_name="Focus Monitor", timeout=timeout)
        except:
            pass
    threading.Thread(target=_fire, daemon=True).start()

# Human-friendly messages (unchanged)
DISTRACTION_MESSAGES = ["Looks like your mind wandered — no worries! Let's refocus. 🎯", "Hey there! The screen misses you. Come back when you're ready. 👀"]
BLINK_MESSAGES = ["Your eyes look tired 😴 — remember to blink and take a short break!", "Eyes feeling heavy? Look away for 20 seconds to recharge. 👁️"]
MILESTONE_MESSAGES = {5: "You've been distracted 5 times. Take a 5-minute break!", 10: "10 distractions logged. Maybe a short walk would help?"}
POSITIVE_MESSAGES = ["Amazing! You've been focused for 5 minutes straight. Keep it up! 🔥"]

_distraction_msg_idx = 0
_blink_msg_idx = 0
_positive_msg_idx = 0

def get_rotating(messages, idx_name):
    global _distraction_msg_idx, _blink_msg_idx, _positive_msg_idx
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
# OUTPUT FILES
# -------------------------------
output_file = "Results/realtime_results_final(1).csv"
graph_dir = "Results/Graphs"
os.makedirs(graph_dir, exist_ok=True)

if not os.path.exists(output_file):
    pd.DataFrame(columns=["Timestamp","EAR","Blink Rate","Gaze Direction","Head Angle","Posture","Emotion","Screen Time","Distraction Count","Focus Level","Prediction"]).to_csv(output_file, index=False)

# -------------------------------
# BUFFERS & DATA COLLECTION
# -------------------------------
score_buffer = deque(maxlen=5)
focus_levels = []
timestamps = []
distraction_events = []

# -------------------------------
# MEDIAPIPE SETUP
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = 468
RIGHT_IRIS = 473

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
focus_streak_start = None
_positive_notified_at = 0
_milestones_notified = set()

print("Starting Real-Time Focus Detection System...")
print("Press 'q' or ESC to exit\n")

# ====================== GRAPH GENERATION FUNCTION ======================
def generate_focus_graph():
    if len(timestamps) < 10:
        return
    plt.figure(figsize=(12, 7), dpi=300)
    gs = GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)

    ax1.plot(timestamps, focus_levels, 'b-', linewidth=2.5, label='Focus Level')
    ax1.set_ylabel('Focus Level (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Focus Level Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    for t, count in distraction_events:
        ax1.axvline(x=t, color='red', alpha=0.4, linestyle='--')
        ax1.text(t, 98, f'D{count}', color='red', fontsize=10, rotation=90)

    ax1.legend()

    if distraction_events:
        times = [t for t, _ in distraction_events]
        counts = [c for _, c in distraction_events]
        ax2.step(times, counts, where='post', color='darkred', linewidth=2)
        ax2.set_ylabel('Cumulative\nDistractions')

    ax2.set_xlabel('Time')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    timestamp_str = session_start_datetime.strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(graph_dir, f"focus_profile_{timestamp_str}.png")
    pdf_path = os.path.join(graph_dir, f"focus_profile_{timestamp_str}.pdf")

    plt.savefig(png_path, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"\nGraph saved: {png_path}")

# -------------------------------
# MAIN LOOP
# -------------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # Reset variables every frame
        gaze = "Away"
        posture = "Straight"
        emotion = "Neutral"
        head_angle = 0.0
        EAR = 0.0
        blink_rate = 0.0
        show_details = True
        warning_text = ""

        if results.multi_face_landmarks:
            if len(results.multi_face_landmarks) > 1:
                warning_text = "MULTIPLE FACES DETECTED!"
                show_details = False
            else:
                landmarks = results.multi_face_landmarks[0].landmark
                x1, y1, x2, y2 = [int(min([lm.x*w for lm in landmarks])), int(min([lm.y*h for lm in landmarks])), 
                                int(max([lm.x*w for lm in landmarks])), int(max([lm.y*h for lm in landmarks]))]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # EAR
                def calculate_EAR(landmarks, eye, w, h):
                    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye]
                    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
                    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
                    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
                    return (A + B) / (2.0 * C) if C != 0 else 0

                leftEAR = calculate_EAR(landmarks, LEFT_EYE, w, h)
                rightEAR = calculate_EAR(landmarks, RIGHT_EYE, w, h)
                EAR = (leftEAR + rightEAR) / 2.0

                if EAR < BLINK_THRESHOLD:
                    blink_frames += 1
                else:
                    if blink_frames >= 2:
                        blink_count += 1
                    blink_frames = 0
                blink_rate = blink_count * 60 / (time.time() - session_start_time + 1)

                # Gaze
                left_iris = landmarks[LEFT_IRIS]
                right_iris = landmarks[RIGHT_IRIS]
                avg_x = (left_iris.x + right_iris.x) / 2
                avg_y = (left_iris.y + right_iris.y) / 2
                gaze = "Screen" if (0.38 < avg_x < 0.62 and 0.35 < avg_y < 0.65) else "Away"

                nose = landmarks[1]
                head_angle = (nose.x - 0.5) * 60
                posture = "Straight" if abs(head_angle) < 12 else "Slouch"
                emotion = np.random.choice(["Happy", "Neutral", "Sad", "Focused"])

        else:
            warning_text = "NO FACE DETECTED!"
            show_details = False

        # Distraction Logic
        current_time = time.time()
        current_datetime = datetime.now()

        if gaze == "Away":
            if distraction_start_time is None:
                distraction_start_time = current_time
                distraction_counted = False
            if (current_time - distraction_start_time >= DISTRACTION_THRESHOLD_SECONDS) and not distraction_counted:
                distraction_count += 1
                distraction_counted = True
                distraction_events.append((current_datetime, distraction_count))
        else:
            distraction_start_time = None
            distraction_counted = False

        # Prediction & Display
        if show_details:
            Eye_Attention_Score = 1 if gaze == "Screen" else 0
            Face_Orientation_Score = 1 if abs(head_angle) < 12 else 0
            Posture_Score = 1 if posture == "Straight" else 0
            emotion_score_map = {"Happy": 1.0, "Neutral": 0.7, "Sad": 0.3, "Focused": 0.9}
            Emotion_Score = emotion_score_map.get(emotion, 0.5)
            Behavior_Score = (Eye_Attention_Score + Face_Orientation_Score + Posture_Score + Emotion_Score) / 4.0

            features = pd.DataFrame([[Eye_Attention_Score, Face_Orientation_Score, Emotion_Score, Posture_Score, Behavior_Score]],
                                  columns=["Eye_Attention_Score", "Face_Orientation_Score", "Emotion_Score", "Posture_Score", "Behavior_Score"])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]

            current_label = "Distracted" if (gaze == "Away" or EAR < 0.18) else ("Focused" if prediction == 1 else "Distracted")
            color = (0, 255, 0) if current_label == "Focused" else (0, 0, 255)

            focus_level = Eye_Attention_Score * 40 + Face_Orientation_Score * 25 + Posture_Score * 15 + Emotion_Score * 20
            if gaze == "Away": focus_level -= 35
            if EAR < 0.2: focus_level -= 25
            focus_level = max(0, min(100, focus_level))

            score_buffer.append(focus_level)
            smooth_focus = int(np.mean(score_buffer))

            if frame_count % 5 == 0:
                timestamps.append(current_datetime)
                focus_levels.append(smooth_focus)
        else:
            current_label = "Warning"
            color = (0, 0, 255)
            smooth_focus = 0

        # On-screen display (your original style)
        if not show_details:
            cv2.putText(frame, warning_text, (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
        else:
            cv2.putText(frame, f"Status: {current_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Focus Level: {smooth_focus}%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, f"Gaze: {gaze} | Distractions: {distraction_count}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Focus Detection System - Real-time Analysis", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

        frame_count += 1

finally:
    cap.release()
    cv2.destroyAllWindows()
    generate_focus_graph()

    print("\nSession Ended. Graph generated successfully!")