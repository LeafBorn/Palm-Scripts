import cv2
import os
import time

# Settings
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

IMAGES_PER_CLASS = 300
DATA_DIR = 'data'

# Create folders
for label in LABELS:
    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Webcam nahi mila!")
    exit()

print("=" * 50)
print("DATA COLLECTION SHURU HOGI")
print("Har letter ke liye haath saamne rakh")
print("SPACE dabao capture shuru karne ke liye")
print("Q dabao quit karne ke liye")
print("=" * 50)

for label in LABELS:
    print(f"\n>>> '{label}' sign ke liye ready ho jao!")
    print(f"    SPACE dabao jab ready ho...")

    # Wait for SPACE
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Sign: {label} | SPACE dabao", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Ready? SPACE dabao!", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.imshow("Data Collection", frame)

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
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(1000)

    # Capture images
    count = 0
    while count < IMAGES_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        img_path = os.path.join(DATA_DIR, label, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1

        cv2.putText(frame, f"Sign: {label} | {count}/{IMAGES_PER_CLASS}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Capturing...", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(50)  # ~20fps

    print(f"    '{label}' done! {count} images save hui.")

cap.release()
cv2.destroyAllWindows()
print("\nData collection complete!")
print("Ab 2_extract_keypoints.py chalao.")