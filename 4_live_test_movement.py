import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

MODEL_DIR = 'movement_model'

# Load model and params
model = tf.keras.models.load_model(f'{MODEL_DIR}/model.keras')
labels = np.load(f'{MODEL_DIR}/labels.npy', allow_pickle=True)
mean = np.load(f'{MODEL_DIR}/mean.npy')
std = np.load(f'{MODEL_DIR}/std.npy')
print(f"Model loaded! Gestures: {labels}")

# MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)

SEQUENCE_LENGTH = 30
sequence = []
prediction_buffer = deque(maxlen=5)
current_gesture = "..."
current_confidence = 0.0

def extract_keypoints(result):
    # Pose
    if result.pose_landmarks:
        pose = np.array([
            [lm.x, lm.y] for lm in result.pose_landmarks.landmark
        ]).flatten()
    else:
        pose = np.zeros(66)

    # Left Hand
    if result.left_hand_landmarks:
        wrist = result.left_hand_landmarks.landmark[0]
        left = np.array([
            [lm.x - wrist.x, lm.y - wrist.y]
            for lm in result.left_hand_landmarks.landmark
        ]).flatten()
    else:
        left = np.zeros(42)

    # Right Hand
    if result.right_hand_landmarks:
        wrist = result.right_hand_landmarks.landmark[0]
        right = np.array([
            [lm.x - wrist.x, lm.y - wrist.y]
            for lm in result.right_hand_landmarks.landmark
        ]).flatten()
    else:
        right = np.zeros(42)

    return np.concatenate([pose, left, right])  # 150 features

cap = cv2.VideoCapture(0)
print("Live test chal raha hai! Q dabao quit karne ke liye.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(img_rgb)

    # Draw landmarks
    mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_draw.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Extract keypoints
    keypoints = extract_keypoints(result)
    keypoints = (keypoints - mean) / std
    sequence.append(keypoints)

    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    # Predict
    if len(sequence) == SEQUENCE_LENGTH:
        input_data = np.expand_dims(sequence, axis=0)  # (1, 30, 150)
        prediction = model.predict(input_data, verbose=0)[0]

        pred_class = np.argmax(prediction)
        confidence = prediction[pred_class]

        if confidence > 0.7:
            prediction_buffer.append(pred_class)
            smooth_pred = max(set(prediction_buffer), key=list(prediction_buffer).count)
            current_gesture = labels[smooth_pred]
            current_confidence = confidence
        else:
            current_gesture = "..."
            current_confidence = 0.0

    # UI
    color = (0, 255, 0) if current_confidence > 0.8 else (0, 165, 255)
    cv2.putText(frame, f"{current_gesture}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 3)

    if current_confidence > 0:
        cv2.putText(frame, f"{current_confidence * 100:.1f}%", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Buffer progress bar
    bar_filled = int(len(sequence) / SEQUENCE_LENGTH * 300)
    cv2.rectangle(frame, (10, h - 40), (310, h - 20), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, h - 40), (10 + bar_filled, h - 20), (0, 255, 255), -1)
    cv2.putText(frame, "Buffer", (315, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.putText(frame, "Q = Quit", (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Movement Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()