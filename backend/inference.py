import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'sign_model')

class SignRecognizer:
    def __init__(self, window_size=30, min_frames=15, conf_threshold=0.55):
        self.window_size = window_size
        self.min_frames = min_frames
        self.conf_threshold = conf_threshold
        
        # Load Model (.h5 prioritized for 100% universal compatibility across Keras/TF versions)
        model_candidates = [
            os.path.join(MODEL_DIR, 'best_model.h5'),
            os.path.join(MODEL_DIR, 'best_model.keras'),
            os.path.join(MODEL_DIR, 'model.keras')
        ]
        
        self.model = None
        for candidate in model_candidates:
            if os.path.exists(candidate):
                try:
                    print(f"📦 Attempting to load model: {candidate}")
                    self.model = tf.keras.models.load_model(candidate)
                    print(f"✅ Successfully loaded model from: {candidate}")
                    break
                except Exception as e:
                    print(f"⚠️ Failed loading {candidate}: {e}")
                    
        if self.model is None:
            raise FileNotFoundError(f"Could not load any valid model from {MODEL_DIR}!")

        self.signs = np.load(os.path.join(MODEL_DIR, 'signs.npy'), allow_pickle=True)
        self.mean = np.load(os.path.join(MODEL_DIR, 'mean.npy'))
        self.std = np.load(os.path.join(MODEL_DIR, 'std.npy'))
        print(f"✅ Model Loaded. Vocabulary: {len(self.signs)} signs.")
        
        # Native MediaPipe Holistic (Ultra-fast C++ backend)
        self.holistic = None
        try:
            mp_holistic = mp.solutions.holistic
            self.holistic = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.4
            )
            print("✅ Native MediaPipe Holistic engine initialized (30+ FPS)!")
        except Exception as e:
            print(f"⚠️ MediaPipe Holistic init warning: {e}")
            try:
                # Fallback import structure
                import mediapipe.python.solutions.holistic as mp_holistic
                self.holistic = mp_holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.4
                )
                print("✅ Native MediaPipe Holistic engine initialized via fallback!")
            except Exception as e2:
                print(f"❌ Could not initialize MediaPipe Holistic: {e2}")
        
        # Rolling Keypoints Buffer
        self.sequence_buffer = deque(maxlen=self.window_size)
        
        # Probability Smoothing Buffer
        self.prediction_history = deque(maxlen=4)
        
        # Speech Lock & State Tracking
        self.active_sign = ""
        self.last_spoken_sign = ""
        self.no_hand_counter = 0

    def extract_keypoints(self, result):
        """Extracts exact 150 landmarks matching 11_record.py and 14_slide_test.py."""
        # Pose: 33 x 2 = 66
        if result and result.pose_landmarks:
            pose = np.array([[lm.x, lm.y] for lm in result.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(66, dtype=np.float32)

        # Left Hand: 21 x 2 = 42 (wrist-relative)
        if result and result.left_hand_landmarks:
            wrist = result.left_hand_landmarks.landmark[0]
            left = np.array([[lm.x - wrist.x, lm.y - wrist.y]
                             for lm in result.left_hand_landmarks.landmark]).flatten()
        else:
            left = np.zeros(42, dtype=np.float32)

        # Right Hand: 21 x 2 = 42 (wrist-relative)
        if result and result.right_hand_landmarks:
            wrist = result.right_hand_landmarks.landmark[0]
            right = np.array([[lm.x - wrist.x, lm.y - wrist.y]
                              for lm in result.right_hand_landmarks.landmark]).flatten()
        else:
            right = np.zeros(42, dtype=np.float32)

        return np.concatenate([pose, left, right]).astype(np.float32)

    def process_frame(self, frame_bgr):
        """Processes BGR webcam frame with native MediaPipe and returns recognition payload."""
        if self.holistic is None:
            return self.process_keypoints(np.zeros(150, dtype=np.float32), False, False)
            
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.holistic.process(img_rgb)
        
        has_left = bool(result.left_hand_landmarks is not None)
        has_right = bool(result.right_hand_landmarks is not None)
        keypoints = self.extract_keypoints(result)
        
        return self.process_keypoints(keypoints, has_left, has_right)

    def process_keypoints(self, keypoints, has_left, has_right):
        """Processes 150 float keypoints through LSTM inference pipeline."""
        has_hand = bool(has_left or has_right)
        
        if has_hand:
            self.no_hand_counter = 0
            self.sequence_buffer.append(keypoints)
        else:
            self.no_hand_counter += 1
            if len(self.sequence_buffer) > 0:
                self.sequence_buffer.popleft()
            if self.no_hand_counter >= 15:
                self.sequence_buffer.clear()
                self.prediction_history.clear()
                self.active_sign = ""
                self.last_spoken_sign = ""
                
        payload = {
            "hand_detected": has_hand,
            "has_left": has_left,
            "has_right": has_right,
            "buffer_length": len(self.sequence_buffer),
            "max_buffer": self.window_size,
            "sign": "",
            "confidence": 0.0,
            "top3": [],
            "trigger_speech": False,
            "audio_file": "",
            "is_new_sign": False,
            "no_hand_frames": self.no_hand_counter,
            "is_locked": bool(self.last_spoken_sign)
        }
        
        # Real-time Prediction
        if has_hand and len(self.sequence_buffer) >= self.min_frames:
            buf = np.array(self.sequence_buffer, dtype=np.float32)
            if len(buf) < self.window_size:
                # Interpolate dynamically up to 30 frames for fast early detection
                indices = np.linspace(0, len(buf) - 1, self.window_size).astype(int)
                seq_arr = buf[indices]
            else:
                seq_arr = buf
                
            norm_seq = (seq_arr - self.mean) / self.std
            input_tensor = tf.convert_to_tensor(np.expand_dims(norm_seq, axis=0), dtype=tf.float32)
            raw_probs = self.model(input_tensor, training=False).numpy()[0]
            
            self.prediction_history.append(raw_probs)
            smooth_probs = np.mean(self.prediction_history, axis=0)
            
            top_indices = np.argsort(smooth_probs)[::-1][:3]
            top1_idx = top_indices[0]
            top1_conf = float(smooth_probs[top1_idx])
            top1_sign = str(self.signs[top1_idx])
            
            payload["confidence"] = round(top1_conf, 3)
            payload["top3"] = [
                {"sign": str(self.signs[i]), "score": round(float(smooth_probs[i]), 3)} 
                for i in top_indices
            ]
            
            if top1_conf >= self.conf_threshold:
                if top1_sign != self.active_sign:
                    payload["is_new_sign"] = True
                    self.active_sign = top1_sign
                payload["sign"] = top1_sign
                
                if top1_sign != self.last_spoken_sign and top1_conf >= (self.conf_threshold + 0.05):
                    payload["trigger_speech"] = True
                    clean_name = "".join([c if c.isalnum() else "_" for c in top1_sign.lower().strip()])
                    payload["audio_file"] = f"{clean_name}.mp3"
                    self.last_spoken_sign = top1_sign
            else:
                payload["sign"] = "Signing..."
        elif has_hand:
            payload["sign"] = self.active_sign if self.active_sign else "Detecting..."
        else:
            payload["sign"] = "READY"
            
        return payload

    def close(self):
        if self.holistic:
            try:
                self.holistic.close()
            except Exception:
                pass
