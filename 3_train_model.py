import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
import os

CSV_PATH = 'keypoints/data.csv'
MODEL_PATH = 'model/model.keras'

os.makedirs('model', exist_ok=True)

# Load data
df = pd.read_csv(CSV_PATH)
print(f"Total samples: {len(df)}")
print(df['label'].value_counts())

X = df.drop('label', axis=1).values.astype(np.float32)
y = df['label'].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nClasses: {le.classes_}")

# Save label encoder classes
np.save('model/classes.npy', le.classes_)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# Normalize input features
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-6
np.save('model/mean.npy', mean)
np.save('model/std.npy', std)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Model - optimized for A-Z (26 classes)
model = keras.Sequential([
    keras.layers.Input(shape=(42,)),

    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(len(le.classes_), activation='softmax')
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
    batch_size=64,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=7, factor=0.5, min_lr=1e-6)
    ]
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc * 100:.2f}%")

# Classification report
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Save model
model.save(MODEL_PATH)
print(f"\nModel saved: {MODEL_PATH}")
print("Ab 4_live_test.py chalao!")