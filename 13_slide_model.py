import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
import os

# ============================================================
# GPU CONFIGURATION
# ============================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Active: {gpus}")
    except RuntimeError as e:
        print(f"⚠️ GPU Config Warning: {e}")
else:
    print("ℹ️ Native Windows TensorFlow runs in CPU mode (TensorFlow dropped native Windows CUDA support in TF 2.11+).")
    print("⚡ Optimized architecture applied for ultra-fast training!")

DATA_DIR = 'processed_data'
MODEL_DIR = 'sign_model'
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
X = np.load(os.path.join(DATA_DIR, 'X.npy'))
y = np.load(os.path.join(DATA_DIR, 'y.npy'))
signs = np.load(os.path.join(DATA_DIR, 'signs.npy'), allow_pickle=True)

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes: {len(signs)}")

# Normalize
mean = X.mean(axis=(0, 1))
std = X.std(axis=(0, 1)) + 1e-6
X = (X - mean) / std

np.save(os.path.join(MODEL_DIR, 'mean.npy'), mean)
np.save(os.path.join(MODEL_DIR, 'std.npy'), std)
np.save(os.path.join(MODEL_DIR, 'signs.npy'), signs)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

NUM_CLASSES = len(signs)
FEATURE_DIM = X.shape[2]
SEQUENCE_LENGTH = X.shape[1]

# Deep High-Accuracy Architecture: Conv1D + BiLSTM + LayerNorm
model = keras.Sequential([
    keras.layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM)),

    # 1️⃣ Spatial & Motion Feature Extraction (Conv1D)
    keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
    keras.layers.LayerNormalization(),
    keras.layers.Dropout(0.2),

    keras.layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'),
    keras.layers.LayerNormalization(),
    keras.layers.Dropout(0.2),

    # 2️⃣ Deep Bidirectional Sequence Learning
    keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
    keras.layers.LayerNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
    keras.layers.LayerNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=False)),
    keras.layers.LayerNormalization(),
    keras.layers.Dropout(0.3),

    # 3️⃣ Deep Multi-Stage Classifier Head
    keras.layers.Dense(512, activation='relu'),
    keras.layers.LayerNormalization(),
    keras.layers.Dropout(0.4),

    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),

    # 4️⃣ Output Layer
    keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train with EarlyStopping & Checkpoint
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=128,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=4,
            factor=0.5,
            min_lr=1e-5,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        )
    ]
)

# Evaluate best model
best_model_path = os.path.join(MODEL_DIR, 'best_model.keras')
best_model = keras.models.load_model(best_model_path)
loss, acc = best_model.evaluate(X_test, y_test)
print(f"\n✅ Best Model Test Accuracy: {acc * 100:.2f}%")

y_pred = np.argmax(best_model.predict(X_test), axis=1)
print("\n", classification_report(y_test, y_pred, target_names=signs))

# Save best model to model.keras as well for consistency
best_model.save(os.path.join(MODEL_DIR, 'model.keras'))
print(f"\n✅ Models successfully saved in '{MODEL_DIR}/':")
print(f"   - best_model.keras")
print(f"   - model.keras")
print(f"   - mean.npy, std.npy, signs.npy")