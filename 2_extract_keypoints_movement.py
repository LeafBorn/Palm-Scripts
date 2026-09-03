import numpy as np
import os

GESTURES = [
    'Hello', 'Thank You', 'Sorry', 'Yes', 'No', 'Good',
    'Please', 'Help', 'I Love You'
]
DATA_DIR = 'movement_data'
OUTPUT_DIR = 'movement_keypoints'

os.makedirs(OUTPUT_DIR, exist_ok=True)

X = []
y = []

for label_idx, gesture in enumerate(GESTURES):
    gesture_dir = os.path.join(DATA_DIR, gesture)
    sequences = os.listdir(gesture_dir)
    success = 0

    for seq_file in sequences:
        seq_path = os.path.join(gesture_dir, seq_file)
        sequence = np.load(seq_path)

        # Shape check = (30, 150)
        if sequence.shape == (30, 150):
            X.append(sequence)
            y.append(label_idx)
            success += 1

    print(f"'{gesture}': {success} sequences loaded")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print(f"\nTotal X shape: {X.shape}")  # (samples, 30, 150)
print(f"Total y shape: {y.shape}")

np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X)
np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y)
np.save(os.path.join(OUTPUT_DIR, 'labels.npy'), np.array(GESTURES))

print(f"\nData saved to {OUTPUT_DIR}/")
print("Ab 3_train_lstm.py chalao.")