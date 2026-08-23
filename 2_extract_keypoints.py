import cv2
import mediapipe as mp
import csv
import os

LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

DATA_DIR = 'data'
OUTPUT_CSV = 'keypoints/data.csv'

os.makedirs('keypoints', exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# 21 landmarks x 2 (x, y) = 42 features
header = [f"{i}_{axis}" for i in range(21) for axis in ['x', 'y']] + ['label']

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for label in LABELS:
        folder = os.path.join(DATA_DIR, label)
        images = os.listdir(folder)
        success = 0
        failed = 0

        for img_name in images:
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            if result.multi_hand_landmarks:
                landmarks = result.multi_hand_landmarks[0]

                # Normalize keypoints relative to wrist (landmark 0)
                wrist = landmarks.landmark[0]
                row = []
                for lm in landmarks.landmark:
                    row.append(lm.x - wrist.x)
                    row.append(lm.y - wrist.y)

                row.append(label)
                writer.writerow(row)
                success += 1
            else:
                failed += 1

        print(f"'{label}': {success} successful, {failed} failed (hand not detected)")


hands.close()
print(f"\nKeypoints saved: {OUTPUT_CSV}")
print("Now run 3_train_model.py.")