import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
import os

DATA_DIR = 'movement_keypoints'
MODEL_DIR = 'movement_model'
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
X = np.load(os.path.join(DATA_DIR, 'X.npy'))
y = np.load(os.path.join(DATA_DIR, 'y.npy'))
labels = np.load(os.path.join(DATA_DIR, 'labels.npy'), allow_pickle=True)

print(f"X shape: {X.shape}")  # (samples, 30, 150)
print(f"y shape: {y.shape}")
print(f"Labels: {labels}")

# Normalize
mean = X.mean(axis=(0, 1))
std = X.std(axis=(0, 1)) + 1e-6
X = (X - mean) / std

np.save(os.path.join(MODEL_DIR, 'mean.npy'), mean)
np.save(os.path.join(MODEL_DIR, 'std.npy'), std)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# LSTM Model - input (30, 150)
model = keras.Sequential([
    keras.layers.Input(shape=(30, 150)),

    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.LSTM(64, return_sequences=False),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=7, factor=0.5, min_lr=1e-6)
    ]
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc * 100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)
print("\n", classification_report(y_test, y_pred, target_names=labels))

model.save(os.path.join(MODEL_DIR, 'model.keras'))
np.save(os.path.join(MODEL_DIR, 'labels.npy'), labels)
print(f"\nModel saved!")
print("Ab 4_live_test_movement.py chalao!")