import numpy as np
import os

# ============================================================
# SETTINGS
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
WINDOW_SIZE = 30    # Frames per sequence
STEP_SIZE = 3       # Sliding window step
DATA_DIR = 'raw_recordings'
OUTPUT_DIR = 'processed_data'
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

X = []
y = []

print("=" * 60)
print("SLIDING WINDOW PROCESSING")
print(f"Window size: {WINDOW_SIZE} frames")
print(f"Step size: {STEP_SIZE} frames")
print("=" * 60)

for label_idx, sign in enumerate(SIGNS):
    sign_dir = os.path.join(DATA_DIR, sign)
    if not os.path.exists(sign_dir):
        print(f"❌ '{sign}' folder nahi mila!")
        continue

    recordings = [f for f in os.listdir(sign_dir) if f.endswith('.npy')]
    total_sequences = 0

    for rec_file in recordings:
        rec_path = os.path.join(sign_dir, rec_file)
        frames = np.load(rec_path)  # (total_frames, 150)

        # Sliding window apply karo
        start = 0
        while start + WINDOW_SIZE <= len(frames):
            window = frames[start:start + WINDOW_SIZE]
            X.append(window)
            y.append(label_idx)
            total_sequences += 1
            start += STEP_SIZE

    print(f"✅ '{sign}': {len(recordings)} recordings → {total_sequences} sequences")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print(f"\nTotal X shape: {X.shape}")
print(f"Total y shape: {y.shape}")
print(f"Sequences per sign avg: {len(X) // len(SIGNS)}")

# Save
np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X)
np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y)
np.save(os.path.join(OUTPUT_DIR, 'signs.npy'), np.array(SIGNS))

print(f"\n✅ Data saved to {OUTPUT_DIR}/")
print("Ab 3_train.py chalao.")