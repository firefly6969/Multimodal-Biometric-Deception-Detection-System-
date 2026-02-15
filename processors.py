"""
Signal processing classes for heart rate estimation and emotion detection.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import threading
import queue
from typing import Optional, Tuple, List
import config


class HeartRateEstimator:
    """
    Heart rate estimator using the trained deep_pulse_model.keras (3D-CNN).
    Buffers 30 face crops via analyze(face_crop); uses rolling average and fallback.
    Calibration (baseline) for deception logic: mean of first CALIBRATION_BPM_SAMPLES valid BPMs.
    """

    CALIBRATION_BPM_SAMPLES = 30

    def __init__(self, model_path: str = "deep_pulse_model.keras"):
        self._dl = DeepHeartRate(model_path=model_path)
        self.baseline_hr: Optional[float] = None
        self.baseline_samples: List[float] = []
        self.calibration_complete = False

    def analyze(self, face_crop: np.ndarray) -> Optional[float]:
        """
        Run DL model on face_crop. Resize 128x128, /255, buffer 30 frames; predict with
        sliding window, rolling average, fallback to last valid. Updates calibration.
        """
        bpm = self._dl.analyze(face_crop)
        if bpm is not None and not self.calibration_complete:
            self.baseline_samples.append(bpm)
            if len(self.baseline_samples) >= self.CALIBRATION_BPM_SAMPLES:
                self.baseline_hr = float(np.mean(self.baseline_samples))
                self.calibration_complete = True
                print(f"Calibration complete. Baseline HR: {self.baseline_hr:.1f} BPM")
        return bpm

    def get_heart_rate(self) -> Optional[float]:
        """Current (smoothed) BPM from the deep model, or None before any valid prediction."""
        return self._dl.get_heart_rate()

    def get_baseline_hr(self) -> Optional[float]:
        """Baseline heart rate from calibration (for deception logic)."""
        return self.baseline_hr


class EmotionDetector:
    """
    Detects emotions from facial images using a custom trained CNN model.
    Runs on a separate thread to prevent blocking the main camera loop.
    """
    
    def __init__(self, model_path='custom_emotion_model.h5'):
        """
        Initialize the emotion detector with a custom trained model.
        
        Args:
            model_path: Path to the saved .h5 model file
        """
        # Load the custom trained model
        try:
            self.model = keras.models.load_model(model_path)
            print(f"Loaded custom emotion model from: {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise
        
        # Emotion class labels (alphabetical order matching model output)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        self.current_emotion = "Neutral"
        self.emotion_queue = queue.Queue()
        self.frame_queue = queue.Queue(maxsize=2)  # Limit queue size
        self.thread = None
        self.running = False
        self.frame_counter = 0
        
    def start(self):
        """Start the emotion detection thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the emotion detection thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def process_frame(self, frame: np.ndarray) -> str:
        """
        Queue a frame for emotion detection (only every Nth frame).
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Current emotion string
        """
        self.frame_counter += 1
        
        # Only process every Nth frame
        if self.frame_counter % config.EMOTION_DETECTION_INTERVAL == 0:
            # Try to add frame to queue (non-blocking)
            try:
                self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass  # Skip if queue is full
        
        # Get latest emotion from queue
        try:
            while True:
                self.current_emotion = self.emotion_queue.get_nowait()
        except queue.Empty:
            pass
        
        return self.current_emotion
    
    def _detection_loop(self):
        """Main loop for emotion detection thread."""
        while self.running:
            try:
                # Get frame from queue (with timeout)
                frame = self.frame_queue.get(timeout=1.0)
                
                # Analyze emotion using custom model
                try:
                    emotion = self.analyze(frame)
                    if emotion:
                        # Put result in queue
                        self.emotion_queue.put(emotion)
                        
                except Exception as e:
                    # Silently handle detection errors
                    pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in emotion detection loop: {e}")
                continue
    
    def analyze(self, face_crop: np.ndarray) -> Optional[str]:
        """
        Analyze emotion from a face crop using the custom trained model.
        
        Args:
            face_crop: Face crop image in BGR format
            
        Returns:
            Detected emotion string, or None if analysis fails
        """
        try:
            # Step 1: Convert BGR to Grayscale
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            
            # Step 2: Resize to (48, 48)
            resized = cv2.resize(gray, (48, 48))
            
            # Step 3: Normalize pixel values (divide by 255.0)
            normalized = resized.astype(np.float32) / 255.0
            
            # Step 4: Reshape to (1, 48, 48, 1) for model input
            input_tensor = normalized.reshape(1, 48, 48, 1)
            
            # Step 5: Run prediction
            predictions = self.model.predict(input_tensor, verbose=0)
            
            # Step 6: Get predicted class index (0-6)
            predicted_class_idx = np.argmax(predictions[0])
            
            # Step 7: Map to emotion label
            emotion = self.emotion_labels[predicted_class_idx]
            
            return emotion
            
        except Exception as e:
            # Silently handle errors
            return None
    
    def get_current_emotion(self) -> str:
        """Get the current detected emotion."""
        return self.current_emotion


class DeepHeartRate:
    """
    Deep learning heart rate estimator using deep_pulse_model.keras.
    Buffers 30 frames via analyze(face_crop): resize 128x128, normalize /255,
    predict when 30 frames; sliding window; rolling average; fallback to last valid.
    """

    CLIP_LEN = 30
    FACE_SIZE = (128, 128)
    HR_MIN, HR_MAX = 40, 200
    ROLLING_SIZE = 5

    def __init__(self, model_path: str = "deep_pulse_model.keras"):
        self.model = keras.models.load_model(model_path)
        self.frame_buffer: List[np.ndarray] = []
        self._raw_bpm_buffer: List[float] = []
        self._last_valid_bpm: Optional[float] = None

    def analyze(self, face_crop: np.ndarray) -> Optional[float]:
        """
        Preprocess face_crop (resize 128x128, /255), add to buffer. When 30 frames:
        predict, sliding window, rolling average. On failure/NaN return previous valid BPM.
        """
        # 1. Resize to (128, 128)
        resized = cv2.resize(face_crop, (self.FACE_SIZE[1], self.FACE_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        # 2. Normalize (divide by 255.0)
        normalized = resized.astype(np.float32) / 255.0
        # 3. Add to buffer
        self.frame_buffer.append(normalized)

        if len(self.frame_buffer) < self.CLIP_LEN:
            return self._last_valid_bpm

        # 4. Convert to (1, 30, 128, 128, 3) and predict
        X = np.stack(self.frame_buffer[: self.CLIP_LEN], axis=0)
        X = X[np.newaxis, ...].astype(np.float32)

        try:
            pred = self.model.predict(X, verbose=0)
            bpm_raw = float(pred[0, 0])
        except Exception:
            return self._last_valid_bpm

        if np.isnan(bpm_raw) or np.isinf(bpm_raw):
            return self._last_valid_bpm

        bpm_raw = float(np.clip(bpm_raw, self.HR_MIN, self.HR_MAX))

        # Rolling average
        self._raw_bpm_buffer.append(bpm_raw)
        if len(self._raw_bpm_buffer) > self.ROLLING_SIZE:
            self._raw_bpm_buffer.pop(0)
        smoothed = float(np.mean(self._raw_bpm_buffer))
        self._last_valid_bpm = smoothed

        # Sliding window: remove oldest frame
        self.frame_buffer.pop(0)
        return self._last_valid_bpm

    def get_heart_rate(self) -> Optional[float]:
        """Return the last valid (smoothed) BPM, or None before any valid prediction."""
        return self._last_valid_bpm
