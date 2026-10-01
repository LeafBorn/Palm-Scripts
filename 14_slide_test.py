import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import time
import os

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_DIR = 'sign_model'
WINDOW_SIZE = 30        # Har prediction 30 frames ke rolling window par hogi
CONF_THRESHOLD = 0.55   # Minimum 55% confidence for positive detection
PREDICT_EVERY_N = 2     # Har 2 frames me predict karein (smooth 30+ FPS webcam)

# ============================================================
# LOAD MODEL & NORMALIZATION PARAMS
# ============================================================
best_model_path = os.path.join(MODEL_DIR, 'best_model.keras')
fallback_model_path = os.path.join(MODEL_DIR, 'model.keras')

if os.path.exists(best_model_path):
    load_path = best_model_path
elif os.path.exists(fallback_model_path):
    load_path = fallback_model_path
else:
    print(f"❌ Koi model file nahi mili in '{MODEL_DIR}/'! Pehle '13_slide_model.py' chala kar model train karein.")
    exit()

print(f"🔄 Loading model from: {load_path}...")
model = tf.keras.models.load_model(load_path)
signs = np.load(os.path.join(MODEL_DIR, 'signs.npy'), allow_pickle=True)
mean = np.load(os.path.join(MODEL_DIR, 'mean.npy'))
std = np.load(os.path.join(MODEL_DIR, 'std.npy'))

print(f"✅ Model Loaded Successfully!")
print(f"✅ Total Signs: {len(signs)}")

# ============================================================
# MEDIAPIPE HOLISTIC SETUP
# ============================================================
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.4
)

def extract_keypoints(result):
    # Pose: 33 x 2 = 66
    if result.pose_landmarks:
        pose = np.array([[lm.x, lm.y] for lm in result.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(66)

    # Left Hand: 21 x 2 = 42 (wrist relative)
    if result.left_hand_landmarks:
        wrist = result.left_hand_landmarks.landmark[0]
        left = np.array([[lm.x - wrist.x, lm.y - wrist.y]
                         for lm in result.left_hand_landmarks.landmark]).flatten()
    else:
        left = np.zeros(42)

    # Right Hand: 21 x 2 = 42 (wrist relative)
    if result.right_hand_landmarks:
        wrist = result.right_hand_landmarks.landmark[0]
        right = np.array([[lm.x - wrist.x, lm.y - wrist.y]
                          for lm in result.right_hand_landmarks.landmark]).flatten()
    else:
        right = np.zeros(42)

    return np.concatenate([pose, left, right])

def hand_detected(result):
    return result.left_hand_landmarks is not None or result.right_hand_landmarks is not None

# ============================================================
# WEBCAM & CONTINUOUS SLIDING BUFFERS
# ============================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera open nahi ho saka! Webcam check karein.")
    exit()

# Buffers
sequence_buffer = deque(maxlen=WINDOW_SIZE)
prediction_history = deque(maxlen=4)   # Smooth averaging over recent predictions
sentence_history = []                  # Confirmed recognized words
last_detected_sign = ""
last_detected_time = 0

current_sign = "Ready..."
current_conf = 0.0
top3_predictions = []

frame_count = 0
prev_time = time.time()
fps = 0.0

print("\n🚀 Live Continuous Sliding Window Test Active! Press 'Q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # FPS Calculation
    curr_time = time.time()
    frame_count += 1
    if curr_time - prev_time >= 1.0:
        fps = frame_count / (curr_time - prev_time)
        frame_count = 0
        prev_time = curr_time

    # Landmark Extraction
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(img_rgb)

    # Draw landmarks
    mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_draw.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    has_hand = hand_detected(result)
    keypoints = extract_keypoints(result)

    # Continuous Rolling Window
    if has_hand:
        sequence_buffer.append(keypoints)
    else:
        # Jab haath nahi hai to buffer slowly reset ho jaye
        if len(sequence_buffer) > 0:
            sequence_buffer.popleft()
        current_sign = "Haath dikhao..."
        current_conf = 0.0
        top3_predictions = []

    # Predict continuously when buffer is full (30 frames)
    if len(sequence_buffer) == WINDOW_SIZE and (frame_count % PREDICT_EVERY_N == 0):
        # Normalize with training mean & std
        seq_array = np.array(sequence_buffer, dtype=np.float32)
        norm_seq = (seq_array - mean) / std
        input_data = np.expand_dims(norm_seq, axis=0)

        # Inference
        probs = model.predict(input_data, verbose=0)[0]
        prediction_history.append(probs)

        # Smoothed Probabilities
        smoothed_probs = np.mean(prediction_history, axis=0)
        top_indices = np.argsort(smoothed_probs)[::-1][:3]
        
        top1_idx = top_indices[0]
        top1_conf = float(smoothed_probs[top1_idx])
        top1_sign = signs[top1_idx]

        top3_predictions = [(signs[idx], float(smoothed_probs[idx])) for idx in top_indices]

        current_conf = top1_conf
        if top1_conf >= CONF_THRESHOLD:
            current_sign = top1_sign
            # Sentence addition with 1.2s cooldown
            if (current_sign != last_detected_sign or (curr_time - last_detected_time > 1.8)) and top1_conf > 0.65:
                sentence_history.append(current_sign)
                if len(sentence_history) > 6:
                    sentence_history.pop(0)
                last_detected_sign = current_sign
                last_detected_time = curr_time
        else:
            current_sign = "Signing..."

    # ============================================================
    # MODERN HUD / USER INTERFACE
    # ============================================================
    
    # 1. Top Status Banner
    banner_color = (30, 30, 30)
    cv2.rectangle(frame, (0, 0), (w, 85), banner_color, -1)

    # Sign Text Color based on Confidence
    if current_conf >= 0.70:
        text_color = (0, 255, 0)      # Green: High confidence
    elif current_conf >= CONF_THRESHOLD:
        text_color = (0, 215, 255)    # Yellow: Moderate
    else:
        text_color = (180, 180, 180)  # Gray: Low/Waiting

    cv2.putText(frame, f"{current_sign.upper()}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, text_color, 3)

    if current_conf > 0:
        cv2.putText(frame, f"{current_conf * 100:.1f}%", (w - 180, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, text_color, 2)

    # 2. Buffer Filling Indicator (Progress Bar under top banner)
    buf_fill = len(sequence_buffer) / WINDOW_SIZE
    bar_width = int(w * buf_fill)
    cv2.rectangle(frame, (0, 80), (bar_width, 85), (0, 255, 0) if buf_fill == 1.0 else (0, 140, 255), -1)

    # 3. Top-3 Candidate Predictions Box (Top-Right)
    if top3_predictions:
        box_x = w - 240
        box_y = 100
        cv2.rectangle(frame, (box_x - 10, box_y - 10), (w - 10, box_y + 80), (20, 20, 20), -1)
        for i, (sign_name, score) in enumerate(top3_predictions):
            line_txt = f"{i+1}. {sign_name}: {score*100:.0f}%"
            c = (0, 255, 0) if i == 0 and score >= CONF_THRESHOLD else (200, 200, 200)
            cv2.putText(frame, line_txt, (box_x, box_y + 20 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1)

    # 4. Bottom Sentence Strip
    cv2.rectangle(frame, (0, h - 45), (w, h), (20, 20, 20), -1)
    sentence_str = "Sentence: " + " ".join(sentence_history)
    cv2.putText(frame, sentence_str, (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 5. Small Footer Info (FPS & Controls)
    hand_status = "Hands: ON" if has_hand else "Hands: OFF"
    cv2.putText(frame, f"{hand_status} | FPS: {fps:.0f} | Buffer: {len(sequence_buffer)}/30 | 'C': Clear | 'Q': Quit",
                (w - 530, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)

    cv2.imshow("Continuous Sign Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence_history = []
        last_detected_sign = ""

cap.release()
cv2.destroyAllWindows()
holistic.close()