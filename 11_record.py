import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ============================================================
# SETTINGS - YAHAN CHANGE KARO
# ============================================================
SIGNS = [
       "hello", "thank you", "sorry", "yes", "no", "please", "help", "love", "good",

"ready", "how are you", "fine", "ok", "great", "hi", "whats-up", "nothing", "busy", "not a lot",

"what's your name", "nice to meet you" , "my name is", "deaf", "hearing", "understand", "I don't understand", "slow", "again", "see you later",

"don't want", "done", "like",  "forget", "i need help",  "more", "go", "happy", "i", "my", "you",

"how", "what", "where", "bathroom", "want", "baby", "brother", "boyfriend", "father", "mother",

"ask", "boy", "girl", "work", "chat", "come", "late", "say", "pay attention", "tell",

"buy", "home", "car","room", "walk", "bread", "cook", "drink", "ice cream", "water", 

"always", "never", "time", "day", "week", "minute", "hour", "year", "what time is it", "see you next week",

"money", "atm", "bank", "deposit", "cash", "price", "free", "earn", "sell", "tax",

"good bye", "take care", "same here", "friend", "introduce", "coffee", "tea", "your name", "small", "medium", "large"
]

"""

"""

RECORDINGS_PER_SIGN = 5      # Har sign ke liye kitni recordings
RECORDING_DURATION = 5       # Seconds per recording
FPS = 30
DATA_DIR = 'raw_recordings'
# ============================================================

# Create folders
for sign in SIGNS:
    os.makedirs(os.path.join(DATA_DIR, sign), exist_ok=True)

# MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)

def extract_keypoints(result):
    # Pose - 33 x 2 = 66
    if result.pose_landmarks:
        pose = np.array([[lm.x, lm.y] for lm in result.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(66)

    # Left Hand - 21 x 2 = 42
    if result.left_hand_landmarks:
        wrist = result.left_hand_landmarks.landmark[0]
        left = np.array([[lm.x - wrist.x, lm.y - wrist.y]
                         for lm in result.left_hand_landmarks.landmark]).flatten()
    else:
        left = np.zeros(42)

    # Right Hand - 21 x 2 = 42
    if result.right_hand_landmarks:
        wrist = result.right_hand_landmarks.landmark[0]
        right = np.array([[lm.x - wrist.x, lm.y - wrist.y]
                          for lm in result.right_hand_landmarks.landmark]).flatten()
    else:
        right = np.zeros(42)

    return np.concatenate([pose, left, right])  # 150 features

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, FPS)

print("=" * 60)
print("SIGN LANGUAGE DATA COLLECTION")
print(f"Signs: {len(SIGNS)}")
print(f"Recordings per sign: {RECORDINGS_PER_SIGN}")
print(f"Duration: {RECORDING_DURATION} sec each")
print(f"Est. total time: {len(SIGNS) * RECORDINGS_PER_SIGN * (RECORDING_DURATION + 3) // 60} min")
print("=" * 60)

for sign in SIGNS:
    print(f"\n>>> Sign: '{sign.upper()}'")
    print(f"    {RECORDINGS_PER_SIGN} recordings karni hain")

    for rec_num in range(RECORDINGS_PER_SIGN):
        print(f"\n    Recording {rec_num + 1}/{RECORDINGS_PER_SIGN}")
        print(f"    SPACE dabao jab ready ho...")

        # Wait for SPACE
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(img_rgb)

            mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_draw.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_draw.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            cv2.putText(frame, f"Sign: {sign.upper()}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(frame, f"Recording {rec_num+1}/{RECORDINGS_PER_SIGN}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.putText(frame, "SPACE = Start | Q = Quit", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            cv2.imshow("Data Collection", frame)

            key = cv2.waitKey(1)
            if key == ord(' '):
                break
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                holistic.close()
                exit()

        # Countdown
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"{i}", (300, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 8)
            cv2.imshow("Data Collection", frame)
            cv2.waitKey(1000)

        # Record
        frames_data = []
        total_frames = RECORDING_DURATION * FPS
        start_time = time.time()

        for frame_num in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(img_rgb)

            mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_draw.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_draw.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            keypoints = extract_keypoints(result)
            frames_data.append(keypoints)

            # Progress bar
            progress = int((frame_num + 1) / total_frames * 400)
            elapsed = time.time() - start_time
            remaining = RECORDING_DURATION - elapsed

            cv2.putText(frame, f"Recording: {sign.upper()}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Time: {remaining:.1f}s", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.rectangle(frame, (10, 110), (410, 130), (60, 60, 60), -1)
            cv2.rectangle(frame, (10, 110), (10 + progress, 130), (0, 255, 0), -1)
            cv2.imshow("Data Collection", frame)
            cv2.waitKey(1)

        # Save recording
        save_path = os.path.join(DATA_DIR, sign, f"rec_{rec_num}.npy")
        np.save(save_path, np.array(frames_data))
        print(f"    Saved! {len(frames_data)} frames")

        time.sleep(0.5)

    print(f"    '{sign}' done!")

cap.release()
cv2.destroyAllWindows()
holistic.close()
print("\n✅ Data collection complete!")
print("Ab 2_sliding_window.py chalao.")