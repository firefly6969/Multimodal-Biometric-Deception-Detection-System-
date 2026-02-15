"""
Main entry point for the Multimodal Deception Detection System.
Handles camera loop, UI rendering, and deception detection logic.
"""

import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from processors import HeartRateEstimator, EmotionDetector
import config
import os
import sys
import platform


class DeceptionDetectionSystem:
    """
    Main system class that orchestrates all three streams:
    - Stream A: Physiological (rPPG Heart Rate)
    - Stream B: Psychological (Emotion Detection)
    - Stream C: Lie Classifier (Heuristic MVP)
    """
    
    def __init__(self):
        # Initialize MediaPipe Face Landmarker (new Tasks API)
        # Use bundled model or download if needed
        model_path = self._get_face_landmarker_model()
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # Initialize processors
        self.hr_estimator = HeartRateEstimator()
        self.emotion_detector = EmotionDetector()
        
        # Camera
        self.cap = None
        
        # State variables
        self.current_hr = None
        self.current_emotion = "Neutral"
        self.deception_alert = False
        self.face_detected = False
        self.face_bbox = None
        
    def _find_available_camera(self) -> int:
        """
        Try to find an available camera by testing different indices.
        
        Returns:
            Camera index if found, raises RuntimeError if none available
        """
        print("Searching for available camera...")
        
        # On macOS, provide specific guidance
        if platform.system() == "Darwin":
            print("Note: On macOS, ensure camera permissions are granted.")
            print("      System Settings > Privacy & Security > Camera > Terminal/Python")
        
        # Try indices 0-4 (common range for most systems)
        for camera_index in range(5):
            print(f"  Trying camera index {camera_index}...", end=" ")
            test_cap = cv2.VideoCapture(camera_index)
            
            if test_cap.isOpened():
                # Give camera a moment to initialize
                time.sleep(0.1)
                # Try to read a frame to confirm it's working
                ret, frame = test_cap.read()
                test_cap.release()
                
                if ret and frame is not None:
                    print(f"✓ Success!")
                    return camera_index
                else:
                    print("✗ Opens but cannot read frames")
            else:
                print("✗ Not available")
        
        # Provide detailed error message
        error_msg = "\nNo camera found. Troubleshooting:\n\n"
        error_msg += "1. Check camera permissions:\n"
        
        if platform.system() == "Darwin":  # macOS
            error_msg += "   - Open System Settings > Privacy & Security > Camera\n"
            error_msg += "   - Enable camera access for Terminal or Python\n"
            error_msg += "   - If using VS Code/Cursor, enable for that app too\n"
        elif platform.system() == "Linux":
            error_msg += "   - Check v4l2 permissions\n"
            error_msg += "   - Try: sudo chmod 666 /dev/video0\n"
        else:  # Windows
            error_msg += "   - Check Windows Privacy Settings > Camera\n"
        
        error_msg += "\n2. Ensure camera is not in use by another application\n"
        error_msg += "3. Try disconnecting and reconnecting the camera\n"
        error_msg += "4. Restart the application after granting permissions\n"
        
        raise RuntimeError(error_msg)
    
    def initialize_camera(self):
        """Initialize the webcam."""
        # Try the configured camera index first
        camera_index = config.CAMERA_INDEX
        self.cap = cv2.VideoCapture(camera_index)
        
        # Test if camera opens and can read frames
        if not self.cap.isOpened():
            print(f"Camera {camera_index} not available, searching for alternatives...")
            self.cap.release()
            camera_index = self._find_available_camera()
            self.cap = cv2.VideoCapture(camera_index)
        
        # Give camera a moment to initialize (especially important on macOS)
        time.sleep(0.2)
        
        # Verify we can actually read from the camera
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            self.cap.release()
            error_msg = "Camera opened but cannot read frames.\n\n"
            error_msg += "Possible causes:\n"
            error_msg += "1. Camera permissions not granted\n"
            if platform.system() == "Darwin":
                error_msg += "   → System Settings > Privacy & Security > Camera\n"
            error_msg += "2. Camera is in use by another application\n"
            error_msg += "3. Camera hardware issue\n"
            error_msg += "\nTry closing other apps that might use the camera and restart."
            raise RuntimeError(error_msg)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        # Get actual resolution (may differ from requested)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized: {actual_width}x{actual_height} @ index {camera_index}")
        
        # Start emotion detection thread
        self.emotion_detector.start()
        
        print("Calibrating baseline heart rate for 10 seconds...")
    
    def _get_face_landmarker_model(self) -> str:
        """
        Get the path to the face landmarker model file.
        Downloads if not present.
        """
        import urllib.request
        
        # Try multiple possible locations
        possible_dirs = [
            os.path.join(os.path.expanduser("~"), ".mediapipe", "models"),
            os.path.join(os.path.dirname(__file__), "models"),
            os.path.join(os.getcwd(), "models")
        ]
        
        model_path = None
        
        # Check if model already exists
        for model_dir in possible_dirs:
            test_path = os.path.join(model_dir, "face_landmarker.task")
            if os.path.exists(test_path):
                model_path = test_path
                print(f"Using existing model at: {model_path}")
                break
        
        # Download if not found
        if model_path is None:
            model_dir = possible_dirs[0]  # Use home directory
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "face_landmarker.task")
            
            if not os.path.exists(model_path):
                print("Downloading face landmarker model (this may take a moment)...")
                model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                try:
                    urllib.request.urlretrieve(model_url, model_path)
                    print(f"Model downloaded successfully to: {model_path}")
                except Exception as e:
                    print(f"Error downloading model: {e}")
                    print("\nPlease download the model manually:")
                    print(f"  URL: {model_url}")
                    print(f"  Save to: {model_path}")
                    raise RuntimeError("Could not download model. Please check internet connection or download manually.")
        
        return model_path
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through all three streams.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Annotated frame with UI overlays
        """
        timestamp = time.time()
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Stream A & B: Face detection and landmark extraction
        detection_result = self.face_landmarker.detect(mp_image)
        
        self.face_detected = False
        self.face_bbox = None
        
        if detection_result.face_landmarks:
            self.face_detected = True
            face_landmarks = detection_result.face_landmarks[0]
            
            # Extract face bounding box from landmarks
            x_coords = [lm.x * w for lm in face_landmarks]
            y_coords = [lm.y * h for lm in face_landmarks]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            self.face_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            # Stream A: Heart Rate Estimation (DL model: 30-frame face crops)
            x, y, w, h = self.face_bbox
            face_crop = frame[y : y + h, x : x + w]
            if face_crop.size > 0:
                bpm = self.hr_estimator.analyze(face_crop)
                if bpm is not None:
                    self.current_hr = bpm
            
            # Stream B: Emotion Detection
            self.current_emotion = self.emotion_detector.process_frame(frame)
            
            # Stream C: Lie Classification
            self._classify_deception()
        
        # Draw UI overlays
        annotated_frame = self._draw_ui(frame)
        
        return annotated_frame
    
    def _classify_deception(self):
        """
        Stream C: Heuristic MVP for deception detection.
        Logic: IF (Current HR > Baseline * 1.15) AND (Emotion is Fear/Nervous) THEN Alert
        """
        self.deception_alert = False
        
        if not self.hr_estimator.calibration_complete:
            return
        
        baseline_hr = self.hr_estimator.get_baseline_hr()
        if baseline_hr is None or self.current_hr is None:
            return
        
        # Check HR threshold
        hr_threshold = baseline_hr * config.HR_THRESHOLD_MULTIPLIER
        hr_elevated = self.current_hr > hr_threshold
        
        # Check emotion
        emotion_lower = self.current_emotion.lower()
        suspicious_emotion = any(emotion in emotion_lower for emotion in config.DECEPTION_EMOTIONS)
        
        # Trigger alert if both conditions met
        if hr_elevated and suspicious_emotion:
            self.deception_alert = True
    
    def _draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw UI overlays on the frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            Frame with UI overlays
        """
        h, w = frame.shape[:2]
        
        # Draw face bounding box
        if self.face_bbox:
            x, y, w_box, h_box = self.face_bbox
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), config.COLOR_GREEN, 2)
        
        # UI Panel Background (semi-transparent)
        panel_height = 150
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, panel_height), config.COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_offset = 35
        
        # Heart Rate Display
        if self.current_hr is not None:
            hr_text = f"HR: {self.current_hr:.1f} BPM"
            hr_color = config.COLOR_RED if self.deception_alert else config.COLOR_GREEN
        else:
            if not self.hr_estimator.calibration_complete:
                hr_text = "HR: Calibrating..."
            else:
                hr_text = "HR: -- BPM"
            hr_color = config.COLOR_YELLOW
        
        cv2.putText(frame, hr_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   config.FONT_SCALE_MEDIUM, hr_color, config.FONT_THICKNESS_MEDIUM)
        
        # Baseline HR (if calibrated)
        if self.hr_estimator.calibration_complete:
            baseline = self.hr_estimator.get_baseline_hr()
            if baseline:
                baseline_text = f"Baseline: {baseline:.1f} BPM"
                cv2.putText(frame, baseline_text, (20, y_offset + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE_SMALL,
                           config.COLOR_WHITE, config.FONT_THICKNESS_SMALL)
        
        # Mood/Emotion Display
        mood_text = f"Mood: {self.current_emotion}"
        cv2.putText(frame, mood_text, (20, y_offset + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE_MEDIUM,
                   config.COLOR_BLUE, config.FONT_THICKNESS_MEDIUM)
        
        # Deception Alert (Big Red Text)
        if self.deception_alert:
            status_text = "STATUS: DECEPTION ALERT"
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX,
                                       config.FONT_SCALE_LARGE, config.FONT_THICKNESS_LARGE)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h - 50
            
            # Draw text with outline for visibility
            cv2.putText(frame, status_text, (text_x - 2, text_y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE_LARGE,
                       config.COLOR_BLACK, config.FONT_THICKNESS_LARGE + 2)
            cv2.putText(frame, status_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE_LARGE,
                       config.COLOR_RED, config.FONT_THICKNESS_LARGE)
        else:
            status_text = "STATUS: TRUTH"
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX,
                                       config.FONT_SCALE_MEDIUM, config.FONT_THICKNESS_MEDIUM)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h - 30
            cv2.putText(frame, status_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE_MEDIUM,
                       config.COLOR_GREEN, config.FONT_THICKNESS_MEDIUM)
        
        # Calibration status
        if not self.hr_estimator.calibration_complete:
            n = len(self.hr_estimator.baseline_samples)
            target = self.hr_estimator.CALIBRATION_BPM_SAMPLES
            calib_text = f"Calibrating... ({n}/{target})"
            cv2.putText(frame, calib_text, (20, y_offset + 75),
                       cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE_SMALL,
                       config.COLOR_YELLOW, config.FONT_THICKNESS_SMALL)
        
        return frame
    
    def run(self):
        """Main execution loop."""
        try:
            self.initialize_camera()
            
            print("\n" + "="*50)
            print("Multimodal Deception Detection System")
            print("="*50)
            print("Press 'q' to quit")
            print("="*50 + "\n")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Process frame
                annotated_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Deception Detection System', annotated_frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        
        if self.cap:
            self.cap.release()
        
        self.emotion_detector.stop()
        
        # Close face landmarker
        if hasattr(self, 'face_landmarker'):
            self.face_landmarker.close()
        
        cv2.destroyAllWindows()
        
        print("Cleanup complete")


def main():
    """Entry point."""
    system = DeceptionDetectionSystem()
    system.run()


if __name__ == "__main__":
    main()
