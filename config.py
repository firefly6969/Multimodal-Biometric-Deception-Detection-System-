"""
Configuration constants for the Multimodal Deception Detection System.
"""

# MediaPipe Face Mesh Forehead ROI Indices
FOREHEAD_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 
    162, 21, 54, 103, 67, 109
]

# Heart Rate Estimation Parameters
HR_BUFFER_SIZE = 150  # Rolling buffer size for HR calculation
HR_FPS = 30  # Target frame rate for interpolation
HR_BANDPASS_LOW = 0.7  # Lower bound of bandpass filter (Hz)
HR_BANDPASS_HIGH = 4.0  # Upper bound of bandpass filter (Hz)
HR_MIN_FREQ = 0.7  # Minimum heart rate frequency (Hz) = 42 BPM
HR_MAX_FREQ = 4.0  # Maximum heart rate frequency (Hz) = 240 BPM

# Calibration Parameters
CALIBRATION_DURATION = 10  # Seconds for baseline HR calculation
CALIBRATION_FPS = 30  # Assumed FPS for calibration
CALIBRATION_FRAMES = CALIBRATION_DURATION * CALIBRATION_FPS  # Total frames for calibration
HR_THRESHOLD_MULTIPLIER = 1.15  # HR must exceed baseline by 15% to trigger alert

# Emotion Detection Parameters
EMOTION_DETECTION_INTERVAL = 6  # Process every Nth frame to reduce lag
DECEPTION_EMOTIONS = ['fear', 'nervous', 'sad', 'angry']  # Emotions that may indicate deception

# UI Colors (BGR format for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_YELLOW = (0, 255, 255)

# UI Display Settings
FONT_SCALE_SMALL = 0.6
FONT_SCALE_MEDIUM = 0.8
FONT_SCALE_LARGE = 1.5
FONT_THICKNESS_SMALL = 1
FONT_THICKNESS_MEDIUM = 2
FONT_THICKNESS_LARGE = 3
LINE_HEIGHT = 30

# Camera Settings
CAMERA_INDEX = 0  # Default webcam index
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
