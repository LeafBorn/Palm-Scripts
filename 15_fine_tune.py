import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
import os

# ============================================================
# SETTINGS
# ============================================================
NEW_SIGNS = ['new_sign1', 'new_sign2', 'new_sign3']  # Naye signs yahan daalo
NEW_DATA_DIR = 'processed_data_new'   # Naye signs ka data
OLD_MODEL_DIR = 'sign_model'          # Purana model
NEW_MODEL_DIR = 'sign_model_updated'  # Updated model save hoga yahan
# ============================================================

os.makedirs(NEW_MODEL_DIR, exist_ok=True)

# Purana model load karo
print("Loading old model...")
old_model = keras.models.load_model(
    os.path.join(OLD_MODEL_DIR, 'best_model.keras')
)

# Purane signs load karo
old_signs = np.load(
    os.path.join(OLD_MODEL_DIR, 'signs.npy'),
    allow_pickle=True
).tolist()

old_mean = np.load(os.path.join(OLD_MODEL_DIR, 'mean.npy'))
old_std = np.load(os.path.join(OLD_MODEL_DIR, 'std.npy'))

print(f"Old signs: {old_signs}")
print(f"New signs: {NEW_SIGNS}")

# All signs combined
all_signs = old_signs + NEW_SIGNS
print(f"Total signs: {len(all_signs)}")

# ============================================================
# Naye signs ka data load karo
# ============================================================
X_new = np.load(os.path.join(NEW_DATA_DIR, 'X.npy'))
y_new_raw = np.load(os.path.join(NEW_DATA_DIR, 'y.npy'))

# y labels update karo — naye signs ka index purane ke baad se shuru hoga
y_new = y_new_raw + len(old_signs)

print(f"\nNew data shape: {X_new.shape}")
print(f"New labels: {np.unique(y_new)}")

# Normalize with old mean/std
X_new_norm = (X_new - old_mean) / old_std

# ============================================================
# Purana data bhi load karo (optional but better accuracy)
# ============================================================
X_old = np.load(os.path.join('processed_data', 'X.npy'))
y_old = np.load(os.path.join('processed_data', 'y.npy'))
X_old_norm = (X_old - old_mean) / old_std

# Combine old + new data
X_combined = np.concatenate([X_old_norm, X_new_norm], axis=0)
y_combined = np.concatenate([y_old, y_new], axis=0)

print(f"\nCombined data shape: {X_combined.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined,
    test_size=0.2, random_state=42, stratify=y_combined
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================
# Model update karo — new output layer add karo
# ============================================================
NUM_OLD_CLASSES = len(old_signs)
NUM_NEW_CLASSES = len(all_signs)

# Purane model ki layers lo (last layer chod ke)
base_input = old_model.input
base_output = old_model.layers[-2].output  # Last Dense se pehle

# Pehle freeze karo purani layers
for layer in old_model.layers[:-4]:  # Last 4 layers trainable rahegi
    layer.trainable = False

# Naya output layer
new_output = keras.layers.Dense(
    NUM_NEW_CLASSES,
    activation='softmax',
    name='new_output'
)(base_output)

# Naya model
new_model = keras.Model(inputs=base_input, outputs=new_output)

new_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Chota LR
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

new_model.summary()

# ============================================================
# Phase 1 — Sirf naye layers train karo
# ============================================================
print("\n🔥 Phase 1: Naye layers train ho rahe hain...")
new_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            os.path.join(NEW_MODEL_DIR, 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        )
    ]
)

# ============================================================
# Phase 2 — Poora model fine tune karo
# ============================================================
print("\n🔥 Phase 2: Full model fine tuning...")

# Sab layers unfreeze karo
for layer in new_model.layers:
    layer.trainable = True

# Bahut chota LR se fine tune
new_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),  # Very small LR
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

new_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=64,
    callbacks=[
        keras.callbacks.ReduceLROnPlateau(
            patience=7, factor=0.5, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(NEW_MODEL_DIR, 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        )
    ]
)

# ============================================================
# Evaluate
# ============================================================
best_model = keras.models.load_model(
    os.path.join(NEW_MODEL_DIR, 'best_model.keras')
)
loss, acc = best_model.evaluate(X_test, y_test)
print(f"\n✅ Updated Model Accuracy: {acc * 100:.2f}%")

y_pred = np.argmax(best_model.predict(X_test), axis=1)
print("\n", classification_report(
    y_test, y_pred,
    target_names=all_signs
))

# Save everything
np.save(os.path.join(NEW_MODEL_DIR, 'signs.npy'), np.array(all_signs))
np.save(os.path.join(NEW_MODEL_DIR, 'mean.npy'), old_mean)
np.save(os.path.join(NEW_MODEL_DIR, 'std.npy'), old_std)
new_model.save(os.path.join(NEW_MODEL_DIR, 'model.keras'))

print(f"\n✅ Updated model saved to {NEW_MODEL_DIR}/")
print(f"   Total signs: {len(all_signs)}")
print("Ab 4_live_test.py mein MODEL_DIR = 'sign_model_updated' karo!")