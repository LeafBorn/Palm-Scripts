import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Settings
GESTURES = [
    'Hello', 'Thank You', 'Sorry', 'Yes', 'No', 'Good',
    'Please', 'Help', 'I Love You', 'Come'
]
SEQUENCES_PER_GESTURE = 100
SEQUENCE_LENGTH = 30
DATA_DIR = 'movement_data'

# Create folders
for gesture in GESTURES:
    os.makedirs(os.path.join(DATA_DIR, gesture), exist_ok=True)

# MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def extract_keypoints(result):
    # Pose - 33 x 2 = 66 features
    if result.pose_landmarks:
        pose = np.array([
            [lm.x, lm.y] for lm in result.pose_landmarks.landmark
        ]).flatten()
    else:
        pose = np.zeros(66)

    # Left Hand - 21 x 2 = 42 features
    if result.left_hand_landmarks:
        wrist = result.left_hand_landmarks.landmark[0]
        left = np.array([
            [lm.x - wrist.x, lm.y - wrist.y]
            for lm in result.left_hand_landmarks.landmark
        ]).flatten()
    else:
        left = np.zeros(42)

    # Right Hand - 21 x 2 = 42 features
    if result.right_hand_landmarks:
        wrist = result.right_hand_landmarks.landmark[0]
        right = np.array([
            [lm.x - wrist.x, lm.y - wrist.y]
            for lm in result.right_hand_landmarks.landmark
        ]).flatten()
    else:
        right = np.zeros(42)

    # Total = 66 + 42 + 42 = 150 features
    return np.concatenate([pose, left, right])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Webcam nahi mila!")
    exit()

print("=" * 50)
print("MOVEMENT DATA COLLECTION - HOLISTIC")
print("Pose + Dono Haath detect hoga")
print("Har gesture ke liye 100 sequences")
print("Har sequence = 30 frames")
print("SPACE dabao shuru karne ke liye")
print("Q dabao quit karne ke liye")
print("=" * 50)

for gesture in GESTURES:
    print(f"\n>>> '{gesture}' gesture ke liye ready ho jao!")
    print(f"    SPACE dabao jab ready ho...")

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

        cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Ready? SPACE dabao!", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.imshow("Data Collection - Holistic", frame)

        key = cv2.waitKey(1)
        if key == ord(' '):
            break
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Countdown
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"{i}...", (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6)
        cv2.imshow("Data Collection - Holistic", frame)
        cv2.waitKey(1000)

    # Capture sequences
    for seq_num in range(SEQUENCES_PER_GESTURE):
        sequence = []

        for frame_num in range(SEQUENCE_LENGTH):
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
            sequence.append(keypoints)

            cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Sequence: {seq_num + 1}/{SEQUENCES_PER_GESTURE}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.putText(frame, f"Frame: {frame_num + 1}/{SEQUENCE_LENGTH}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

            progress = int((frame_num + 1) / SEQUENCE_LENGTH * 400)
            cv2.rectangle(frame, (10, 150), (410, 170), (60, 60, 60), -1)
            cv2.rectangle(frame, (10, 150), (10 + progress, 170), (0, 255, 0), -1)

            cv2.imshow("Data Collection - Holistic", frame)
            cv2.waitKey(1)

        save_path = os.path.join(DATA_DIR, gesture, f"{seq_num}.npy")
        np.save(save_path, np.array(sequence))
        time.sleep(0.5)

    print(f"    '{gesture}' done! {SEQUENCES_PER_GESTURE} sequences saved.")

cap.release()
cv2.destroyAllWindows()
holistic.close()
print("\nData collection complete!")
print("Ab 2_extract_keypoints_movement.py chalao.")