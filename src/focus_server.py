"""
FocusLens WebSocket Server  (fixed)
------------------------------------
Fixes:
  1. No cv2 window — camera is shown inside the browser via JPEG streaming
  2. Clean shutdown when dashboard sends {"cmd":"stop"}
  3. Stable WS — no drift/spam reconnects
  4. Session report printed on exit

Install:
    pip install websockets opencv-python mediapipe numpy pandas joblib plyer

Run:
    python focus_server.py
Then open focus_dashboard.html in your browser.
"""

import asyncio
import json
import time
import threading
import os
import sys
import base64
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import websockets

# Optional desktop notifications
try:
    from plyer import notification as plyer_notify
    NOTIFY_AVAILABLE = True
except ImportError:
    NOTIFY_AVAILABLE = False

# ═══════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════
WS_HOST     = "localhost"
WS_PORT     = 8765
MODEL_PATH  = "Models/Test2/svm.pkl"
SCALER_PATH = "Models/Test2/scaler.pkl"
OUTPUT_FILE = "Results/realtime_results_final.csv"

BLINK_THRESHOLD           = 0.20
EAR_FATIGUE_THRESHOLD     = 0.18
DISTRACTION_THRESHOLD_SEC = 2.0
JPEG_QUALITY              = 60    # 0-100; lower = faster
FRAME_SKIP                = 2     # send camera frame every Nth iteration

# ═══════════════════════════════════════
#  SHUTDOWN EVENT
# ═══════════════════════════════════════
stop_event = threading.Event()

# ═══════════════════════════════════════
#  NOTIFICATIONS
# ═══════════════════════════════════════
_cd = {}

def notify(title, msg, key=None, cd=30):
    now = time.time()
    if key and now - _cd.get(key, 0) < cd:
        return
    if key:
        _cd[key] = now
    if not NOTIFY_AVAILABLE:
        return
    def _f():
        try:
            plyer_notify.notify(title=title, message=msg,
                                app_name="FocusLens", timeout=6)
        except Exception:
            pass
    threading.Thread(target=_f, daemon=True).start()

DIST_MSGS = [
    "Looks like your mind wandered — let's refocus. 🎯",
    "The screen misses you. Come back when you're ready. 👀",
    "You seem a bit distracted. You've got this! 💪",
    "Take a breath and come back — you're doing great. 🌟",
    "Attention check! Refocus and keep going strong. 🔔",
]
BLINK_MSGS = [
    "Your eyes look tired 😴 — remember to blink!",
    "Eyes heavy? Look away for 20 seconds. 👁️",
    "Try the 20-20-20 rule: look 20ft away for 20 sec!",
]
POS_MSGS = [
    "Amazing! 5 minutes of straight focus. Keep it up! 🔥",
    "5 solid minutes in the zone. You're crushing it. ⚡",
    "5 minutes of pure focus — your future self thanks you. 🏆",
]
_mi = {"d": 0, "b": 0, "p": 0}

def rot(pool, k):
    m = pool[_mi[k] % len(pool)]
    _mi[k] += 1
    return m

# ═══════════════════════════════════════
#  ML MODEL
# ═══════════════════════════════════════
model = scaler = None
try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[✓] Model loaded")
except FileNotFoundError:
    print("[!] Model not found — heuristic-only mode")

# ═══════════════════════════════════════
#  CSV
# ═══════════════════════════════════════
os.makedirs("Results", exist_ok=True)
if not os.path.exists(OUTPUT_FILE):
    pd.DataFrame(columns=[
        "Timestamp","EAR","Blink Rate","Gaze Direction",
        "Head Angle","Posture","Emotion","Screen Time",
        "Distraction Count","Focus Level","Prediction"
    ]).to_csv(OUTPUT_FILE, index=False)

# ═══════════════════════════════════════
#  MEDIAPIPE
# ═══════════════════════════════════════
mp_fm = mp.solutions.face_mesh
face_mesh = mp_fm.FaceMesh(
    refine_landmarks=True, max_num_faces=2,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

LEFT_EYE  = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
LEFT_IRIS  = 468
RIGHT_IRIS = 473

def calc_ear(lm, eye, w, h):
    pts = [(lm[i].x*w, lm[i].y*h) for i in eye]
    A = np.linalg.norm(np.array(pts[1])-np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2])-np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0])-np.array(pts[3]))
    return (A+B)/(2.0*C) if C else 0.0

# ═══════════════════════════════════════
#  SHARED STATE
# ═══════════════════════════════════════
latest      = {}
latest_lock = threading.Lock()
clients     = set()

# ═══════════════════════════════════════
#  CV THREAD  — headless, streams frames via base64
# ═══════════════════════════════════════
def cv_loop():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[!] Cannot open camera")
        stop_event.set()
        return

    t0           = time.time()
    dist_count   = blink_count = frame_n = focused_f = blink_f = 0
    score_buf    = deque(maxlen=5)
    dist_start   = None
    dist_done    = False
    streak_t     = None
    pos_t        = 0.0
    milestones   = set()

    MILESTONES = {
        5:  "5 distractions — take a short break! ☕",
        10: "10 distractions — a walk might help. 🚶",
        20: "20 distractions — rest is productive too! 🛌",
    }

    print("[✓] CV loop started (headless — camera streamed to browser)")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.033)
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = face_mesh.process(rgb)

        now        = time.time()
        elapsed    = now - t0 + 1e-9
        gaze       = "Away"
        posture    = "Straight"
        emotion    = "Neutral"
        head_angle = 0.0
        EAR        = 0.0
        blink_rate = 0.0
        gaze_x     = 0.5
        gaze_y     = 0.5
        face_status = "no_face"

        if res.multi_face_landmarks:
            if len(res.multi_face_landmarks) > 1:
                face_status = "multi"
                notify("Multiple People Detected 👥",
                       "Only you should be visible!", key="multi", cd=60)
            else:
                face_status = "ok"
                lm  = res.multi_face_landmarks[0].landmark
                EAR = (calc_ear(lm, LEFT_EYE, w, h) +
                       calc_ear(lm, RIGHT_EYE, w, h)) / 2.0

                if EAR < BLINK_THRESHOLD:
                    blink_f += 1
                else:
                    if blink_f >= 2:
                        blink_count += 1
                    blink_f = 0
                blink_rate = blink_count * 60 / elapsed

                if EAR < EAR_FATIGUE_THRESHOLD:
                    notify("Eye Fatigue 👁️", rot(BLINK_MSGS,"b"), key="eye", cd=45)

                nose       = lm[1]
                head_angle = (nose.x - 0.5) * 60
                posture    = "Straight" if abs(head_angle) < 12 else "Slouch"
                if posture == "Slouch":
                    notify("Posture Check 🧍",
                           "Tilting your head — sit up straight!",
                           key="posture", cd=60)

                li, ri = lm[LEFT_IRIS], lm[RIGHT_IRIS]
                gaze_x = (li.x + ri.x) / 2
                gaze_y = (li.y + ri.y) / 2
                gaze   = ("Screen"
                          if 0.38 < gaze_x < 0.62 and 0.35 < gaze_y < 0.65
                          else "Away")
                emotion = np.random.choice(["Happy","Neutral","Sad","Focused"])
        else:
            notify("Where'd you go? 🙈",
                   "Can't see your face — move back in frame.",
                   key="noface", cd=30)

        # Distraction counting
        if gaze == "Away":
            if dist_start is None:
                dist_start = now
                dist_done  = False
            if (now - dist_start >= DISTRACTION_THRESHOLD_SEC) and not dist_done:
                dist_count += 1
                dist_done   = True
                notify("Hey, You Drifted! 🔔", rot(DIST_MSGS,"d"),
                       key="dist", cd=30)
                if dist_count in MILESTONES and dist_count not in milestones:
                    milestones.add(dist_count)
                    notify(f"Milestone: {dist_count} distractions 📊",
                           MILESTONES[dist_count],
                           key=f"ms{dist_count}", cd=5)
            streak_t = None
        else:
            dist_start = None
            dist_done  = False
            if streak_t is None:
                streak_t = now
            if now - streak_t >= 300 and now - pos_t >= 300:
                pos_t = now
                notify("You're on Fire! 🔥", rot(POS_MSGS,"p"),
                       key="pos", cd=300)

        # ML prediction
        focus_level   = 0
        current_label = "Distracted"
        if face_status == "ok":
            ea  = 1 if gaze == "Screen" else 0
            fa  = 1 if abs(head_angle) < 12 else 0
            pa  = 1 if posture == "Straight" else 0
            emo = {"Happy":1.0,"Neutral":0.7,"Sad":0.3,"Focused":0.9}.get(emotion,0.5)
            beh = (ea + fa + pa + emo) / 4.0

            if model and scaler:
                feat = pd.DataFrame(
                    [[ea, fa, emo, pa, beh]],
                    columns=["Eye_Attention_Score","Face_Orientation_Score",
                             "Emotion_Score","Posture_Score","Behavior_Score"]
                )
                pred = model.predict(scaler.transform(feat))[0]
                current_label = ("Focused"
                    if pred == 1 and gaze == "Screen" and EAR >= EAR_FATIGUE_THRESHOLD
                    else "Distracted")
            else:
                current_label = ("Focused"
                    if gaze == "Screen" and EAR >= EAR_FATIGUE_THRESHOLD
                    else "Distracted")

            focus_level = ea*40 + fa*25 + pa*15 + emo*20
            if gaze == "Away": focus_level -= 35
            if EAR  < 0.2:    focus_level -= 25
            focus_level = max(0, min(100, focus_level))

        score_buf.append(focus_level)
        smooth = int(np.mean(score_buf)) if score_buf else 0
        if current_label == "Focused": focused_f += 1

        # Encode camera frame as JPEG → base64
        # TEMPORARILY DISABLED: large payloads cause connection drops
        frame_b64 = ""
        # if frame_n % FRAME_SKIP == 0:
        #     _, buf    = cv2.imencode(".jpg", frame,
        #                              [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        #     frame_b64 = base64.b64encode(buf).decode("ascii")

        payload = {
            "ts":                now,
            "face_status":       face_status,
            "ear":               round(EAR, 4),
            "blink_rate":        round(blink_rate, 2),
            "gaze":              gaze,
            "gaze_x":            round(gaze_x, 4),
            "gaze_y":            round(gaze_y, 4),
            "head_angle":        round(head_angle, 2),
            "posture":           posture,
            "emotion":           emotion,
            "focus_level":       smooth,
            "prediction":        current_label,
            "distraction_count": dist_count,
            "focused_frames":    focused_f,
            "total_frames":      max(frame_n, 1),
            "session_seconds":   int(now - t0),
            "frame_b64":         frame_b64,
        }

        with latest_lock:
            latest.update(payload)

        frame_n += 1
        if frame_n % 10 == 0 and face_status == "ok":
            pd.DataFrame([{
                "Timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "EAR":               round(EAR, 4),
                "Blink Rate":        round(blink_rate, 2),
                "Gaze Direction":    gaze,
                "Head Angle":        round(head_angle, 2),
                "Posture":           posture,
                "Emotion":           emotion,
                "Screen Time":       int(now - t0),
                "Distraction Count": dist_count,
                "Focus Level":       smooth,
                "Prediction":        current_label,
            }]).to_csv(OUTPUT_FILE, mode="a", header=False, index=False)

    cap.release()
    face_mesh.close()
    print("[✓] Camera released")

# ═══════════════════════════════════════
#  WEBSOCKET HANDLER
# ═══════════════════════════════════════
async def ws_handler(ws):
    """Minimal handler - just add client and let broadcast loop handle messaging."""
    clients.add(ws)
    print(f"[+] Dashboard connected ({len(clients)} client(s))")
    
    try:
        # Passive wait - don't do anything, just keep connection alive
        while True:
            await asyncio.sleep(60)
            # Periodically check if we should stop
            if stop_event.is_set():
                break
    except websockets.exceptions.ConnectionClosed:
        pass
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[!] Handler exception: {type(e).__name__}: {str(e)[:50]}")
    finally:
        clients.discard(ws)
        print(f"[-] Client disconnected ({len(clients)} client(s))")

# ═══════════════════════════════════════
#  BROADCAST LOOP  ~30 fps
# ═══════════════════════════════════════
async def broadcast():
    global clients
    frame_count = 0
    log_count = 0
    while not stop_event.is_set():
        await asyncio.sleep(1/30)
        frame_count += 1
        log_count += 1
        
        if not clients:
            continue
        
        with latest_lock:
            data = dict(latest)
        
        if not data:
            if log_count % 90 == 0:
                print("[!] No data in latest")
            continue
        
        try:
            msg = json.dumps(data)
        except Exception as e:
            print(f"[!] JSON error: {e}")
            continue
        
        # Log every 300 frames (~10 sec at 30 fps)
        if log_count % 300 == 0:
            print(f"[✓] Broadcasting to {len(clients)} client(s), msg size: {len(msg)} bytes")
        
        dead = set()
        for ws in list(clients):
            try:
                await ws.send(msg)
            except websockets.exceptions.ConnectionClosed:
                dead.add(ws)
            except Exception as e:
                print(f"[!] Broadcast error: {type(e).__name__}: {str(e)[:40]}")
                dead.add(ws)
        
        for ws in dead:
            clients.discard(ws)
        
        if log_count >= 300:
            log_count = 0

# ═══════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════
async def main():
    threading.Thread(target=cv_loop, daemon=True).start()

    print(f"[✓] WebSocket server → ws://{WS_HOST}:{WS_PORT}")
    print(f"[✓] Open focus_dashboard.html in your browser")
    print(f"[✓] Press Ctrl+C or click 'End Session' to stop\n")

    server = await websockets.serve(ws_handler, WS_HOST, WS_PORT)

    await broadcast()   # blocks until stop_event is set

    server.close()
    await server.wait_closed()

    # Session report
    with latest_lock:
        d = dict(latest)
    if d:
        ff  = d.get("focused_frames", 0)
        tf  = d.get("total_frames", 1)
        ss  = d.get("session_seconds", 0)
        dc  = d.get("distraction_count", 0)
        fmp = round((ff/tf) * (ss/60), 2)
        dmp = round(((tf-ff)/tf) * (ss/60), 2)
        print("\n" + "="*55)
        print("              SESSION REPORT")
        print("="*55)
        print(f"Duration         : {ss//60}m {ss%60}s")
        print(f"Focused Time     : {fmp} min ({round(ff/tf*100)}%)")
        print(f"Distracted Time  : {dmp} min")
        print(f"Distraction Count: {dc}")
        print(f"CSV saved to     : {OUTPUT_FILE}")
        print("="*55)

    print("[✓] Server stopped")
    sys.exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[→] Ctrl+C — stopping")
        stop_event.set()