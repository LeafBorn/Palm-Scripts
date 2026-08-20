import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

MODEL_PATH = 'model/model.keras'
CLASSES_PATH = 'model/classes.npy'
MEAN_PATH = 'model/mean.npy'
STD_PATH = 'model/std.npy'

# Load model, classes and normalization params
model = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)
mean = np.load(MEAN_PATH)
std = np.load(STD_PATH)
print(f"Model loaded! Classes: {classes}")

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Smoothing predictions
prediction_buffer = deque(maxlen=15)

cap = cv2.VideoCapture(0)
print("Live test chal raha hai! Q dabao quit karne ke liye.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    label_text = "Haath dikhao..."
    confidence = 0.0

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract keypoints
        wrist = landmarks.landmark[0]
        row = []
        for lm in landmarks.landmark:
            row.append(lm.x - wrist.x)
            row.append(lm.y - wrist.y)

        # Normalize
        input_data = np.array(row, dtype=np.float32)
        input_data = (input_data - mean) / std
        input_data = input_data.reshape(1, -1)

        prediction = model.predict(input_data, verbose=0)[0]
        pred_class = np.argmax(prediction)
        confidence = prediction[pred_class]

        # Smoothing
        prediction_buffer.append(pred_class)
        smooth_pred = max(set(prediction_buffer), key=list(prediction_buffer).count)

        label_text = f"{classes[smooth_pred]} ({confidence * 100:.1f}%)"

        # Color based on confidence
        color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)
        cv2.putText(frame, label_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
    else:
        prediction_buffer.clear()
        cv2.putText(frame, label_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, "Q = Quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imshow("Hand Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()