import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
try:
    import mediapipe.python.solutions.holistic as mp_holistic
except Exception:
    try:
        from mediapipe import solutions
        mp_holistic = solutions.holistic
    except Exception:
        mp_holistic = getattr(mp, 'solutions', None)
        if mp_holistic:
            mp_holistic = mp_holistic.holistic
from collections import deque
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'sign_model')

class SignRecognizer:
    def __init__(self, window_size=30, min_frames=25, conf_threshold=0.70, margin_threshold=0.15, consensus_frames=3, history_len=5):
        self.window_size = window_size
        self.min_frames = min_frames
        self.conf_threshold = conf_threshold
        self.margin_threshold = margin_threshold
        self.consensus_frames = consensus_frames
        self.history_len = history_len
        
        # Load Model (HDF5 .h5 prioritized for 100% universal Linux/Cloud compatibility)
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
        
        # MediaPipe Holistic - Higher sensitivity for reliable hand detection
        print("📷 Initializing MediaPipe Holistic...")
        self.mp_holistic = mp_holistic
        try:
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.35,
                min_tracking_confidence=0.25
            )
            print("✅ MediaPipe Holistic initialized successfully!")
        except Exception as e:
            print(f"⚠️ Retrying MediaPipe Holistic with model_complexity=0: {e}")
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=False,
                min_detection_confidence=0.35,
                min_tracking_confidence=0.25
            )
            print("✅ MediaPipe Holistic (complexity=0 fallback) initialized!")
        
        # Rolling Keypoints Buffer
        self.sequence_buffer = deque(maxlen=self.window_size)
        
        # Probability & Consensus Buffers
        self.prob_history = deque(maxlen=4)
        self.sign_history = deque(maxlen=self.history_len)
        
        # Speech Lock & State Tracking
        self.active_confirmed_sign = ""
        self.last_spoken_sign = ""
        self.no_hand_counter = 0
        self.no_hand_reset_threshold = 12  # Resilient: only flush buffer if hands drop for ~0.4s

    def extract_keypoints(self, result):
        # Pose - 33 x 2 = 66
        if result.pose_landmarks:
            pose = np.array([[lm.x, lm.y] for lm in result.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(66)

        # Left Hand - 21 x 2 = 42
        if result.left_hand_landmarks:
            wrist = result.left_hand_landmarks.landmark[0]
            left = np.array([[lm.x - wrist.x, lm.y - wrist.y]
                             for lm in result.left_hand_landmarks.landmark]).flatten()
        else:
            left = np.zeros(42)

        # Right Hand - 21 x 2 = 42
        if result.right_hand_landmarks:
            wrist = result.right_hand_landmarks.landmark[0]
            right = np.array([[lm.x - wrist.x, lm.y - wrist.y]
                              for lm in result.right_hand_landmarks.landmark]).flatten()
        else:
            right = np.zeros(42)

        return np.concatenate([pose, left, right])

    def process_frame(self, frame_bgr):
        """Processes a single BGR video frame and returns filtered, consensus-backed prediction payload."""
        # Flip frame horizontally to match training orientation
        frame_bgr = cv2.flip(frame_bgr, 1)
        h, w, _ = frame_bgr.shape
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.holistic.process(img_rgb)
        
        has_left = result.left_hand_landmarks is not None
        has_right = result.right_hand_landmarks is not None
        has_hand = has_left or has_right
        
        keypoints = self.extract_keypoints(result)
        
        # 1. SMART BUFFER & HAND TRANSITION MANAGEMENT
        if has_hand:
            self.no_hand_counter = 0
            self.sequence_buffer.append(keypoints)
        else:
            self.no_hand_counter += 1
            # If tracking was temporarily lost for 1-5 frames during hand motion,
            # carry forward the last valid frame so the buffer doesn't stall at 10-15 frames!
            if self.no_hand_counter < 6 and len(self.sequence_buffer) > 0:
                self.sequence_buffer.append(self.sequence_buffer[-1])
            elif self.no_hand_counter >= self.no_hand_reset_threshold:
                # Hand truly dropped down: flush buffers & unlock speech for next gesture
                self.sequence_buffer.clear()
                self.prob_history.clear()
                self.sign_history.clear()
                self.active_confirmed_sign = ""
                self.last_spoken_sign = ""  # Unlocked! Next time hand appears, can speak immediately
                
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
        
        # 2. INSTANT INFERENCE (FROM 5 FRAMES UPWARDS VIA TEMPORAL INTERPOLATION)
        if has_hand and len(self.sequence_buffer) >= self.min_frames:
            buf = np.array(self.sequence_buffer, dtype=np.float32)
            if len(buf) < self.window_size:
                # Interpolate / stretch current frames up to 30 frames for 0ms cold-start latency!
                indices = np.linspace(0, len(buf) - 1, self.window_size).astype(int)
                seq_arr = buf[indices]
            else:
                seq_arr = buf
            
            norm_seq = (seq_arr - self.mean) / self.std
            
            # Ultra-fast direct call (avoids slow model.predict overhead)
            input_tensor = tf.convert_to_tensor(np.expand_dims(norm_seq, axis=0), dtype=tf.float32)
            raw_probs = self.model(input_tensor, training=False).numpy()[0]
            
            self.prob_history.append(raw_probs)
            smooth_probs = np.mean(self.prob_history, axis=0)
            
            top_indices = np.argsort(smooth_probs)[::-1][:3]
            top1_idx = top_indices[0]
            top2_idx = top_indices[1] if len(top_indices) > 1 else top1_idx
            
            top1_conf = float(smooth_probs[top1_idx])
            top2_conf = float(smooth_probs[top2_idx])
            margin = top1_conf - top2_conf
            candidate_sign = str(self.signs[top1_idx])
            
            payload["confidence"] = round(top1_conf, 3)
            payload["top3"] = [
                {"sign": str(self.signs[i]), "score": round(float(smooth_probs[i]), 3)} 
                for i in top_indices
            ]
            
            # 3. CONFIDENCE & MARGIN FILTER
            if top1_conf >= self.conf_threshold and margin >= self.margin_threshold:
                self.sign_history.append(candidate_sign)
            else:
                self.sign_history.append("UNCERTAIN")
                
            # 4. INSTANT CONSENSUS CHECK
            candidate_count = self.sign_history.count(candidate_sign)
            req_consensus = 2 if (len(self.sequence_buffer) < 15 or top1_conf >= 0.72) else self.consensus_frames
            
            if candidate_count >= req_consensus and candidate_sign != "UNCERTAIN":
                # Consistent, verified prediction
                if candidate_sign != self.active_confirmed_sign:
                    payload["is_new_sign"] = True
                    self.active_confirmed_sign = candidate_sign
                payload["sign"] = candidate_sign
                
                # 5. STRICT SPEECH LOCK (SINGLE-FIRE LATCH)
                if candidate_sign != self.last_spoken_sign:
                    payload["trigger_speech"] = True
                    clean_name = "".join([c if c.isalnum() else "_" for c in candidate_sign.lower().strip()])
                    payload["audio_file"] = f"{clean_name}.mp3"
                    self.last_spoken_sign = candidate_sign  # Lock until sign changes or hand leaves
            else:
                if self.active_confirmed_sign:
                    payload["sign"] = self.active_confirmed_sign
                elif top1_conf >= self.conf_threshold:
                    payload["sign"] = candidate_sign
                    
        elif has_hand:
            payload["sign"] = self.active_confirmed_sign if self.active_confirmed_sign else ""
        else:
            payload["sign"] = "READY"
            
        return payload

    def close(self):
        self.holistic.close()
