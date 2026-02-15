"""
UBFC-rPPG Data Loader: keras.utils.Sequence, open-process-close per batch.
Memory-safe: does NOT load all videos into RAM.
Dataset root: ./datasets/ubfc/data_folder (subject1, subject2, ... with vid.avi and ground_truth.txt).
"""

import os
import glob
import cv2
import numpy as np
from scipy import signal
from typing import List, Tuple, Optional, Union
import tensorflow as tf
from tensorflow import keras

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

CLIP_LEN = 30
FACE_SIZE = (128, 128)
UBFC_DEFAULT_FPS = 30.0
UBFC_PPG_FS = 60.0
HR_BPM_MIN, HR_BPM_MAX = 40, 200

# Default path to UBFC subject folders
DEFAULT_DATA_FOLDER = "./datasets/ubfc/data_folder"


def discover_subject_folders(root: str) -> List[str]:
    """
    Find subject folders under root (e.g. ./datasets/ubfc/data_folder) that contain
    vid.avi and ground_truth.txt. Returns list of subject directory paths.
    """
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        subj = os.path.join(root, name)
        if not os.path.isdir(subj):
            continue
        if _find_video(subj) and _find_ground_truth(subj):
            out.append(subj)
    return out


def _find_video(subj: str) -> Optional[str]:
    for v in ("vid.avi", "video.avi", "0.avi", "1.avi"):
        p = os.path.join(subj, v)
        if os.path.isfile(p):
            return p
    for pat in (os.path.join(subj, "*.avi"), os.path.join(subj, "*.mp4")):
        cand = glob.glob(pat)
        if cand:
            return cand[0]
    return None


def _find_ground_truth(subj: str) -> Optional[str]:
    for g in ("ground_truth.txt", "gt.txt", "heartrate.txt"):
        p = os.path.join(subj, g)
        if os.path.isfile(p):
            return p
    for pat in (os.path.join(subj, "*ground*.txt"), os.path.join(subj, "*gt*.txt")):
        cand = glob.glob(pat)
        if cand:
            return cand[0]
    return None


def _get_face_landmarker_model() -> str:
    import urllib.request
    for d in (
        os.path.join(os.path.expanduser("~"), ".mediapipe", "models"),
        os.path.join(os.path.dirname(__file__), "models"),
        os.path.join(os.getcwd(), "models"),
    ):
        p = os.path.join(d, "face_landmarker.task")
        if os.path.exists(p):
            return p
    d = os.path.join(os.path.expanduser("~"), ".mediapipe", "models")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "face_landmarker.task")
    if not os.path.exists(p):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        try:
            urllib.request.urlretrieve(url, p)
        except Exception as e:
            raise RuntimeError(f"Download face_landmarker.task failed: {e}. Save to {p}")
    return p


def _build_face_detector():
    if not _MEDIAPIPE_AVAILABLE:
        raise RuntimeError("MediaPipe required. pip install mediapipe")
    base = mp_tasks.BaseOptions(model_asset_path=_get_face_landmarker_model())
    opt = vision.FaceLandmarkerOptions(
        base_options=base,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(opt)


def _detect_face_bbox(frame_bgr: np.ndarray, face_landmarker) -> Optional[Tuple[int, int, int, int]]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    try:
        img_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = face_landmarker.detect(img_mp)
    except Exception:
        return None
    if not res.face_landmarks:
        return None
    h, w = frame_bgr.shape[:2]
    lms = res.face_landmarks[0]
    xs = [p.x * w for p in lms]
    ys = [p.y * h for p in lms]
    x1 = int(max(0, min(xs) - 10))
    x2 = int(min(w, max(xs) + 10))
    y1 = int(max(0, min(ys) - 10))
    y2 = int(min(h, max(ys) + 10))
    return (x1, y1, x2 - x1, y2 - y1)


def _floats_from_line(ln: str) -> List[float]:
    out = []
    for p in ln.replace(",", " ").split():
        try:
            out.append(float(p))
        except ValueError:
            pass
    return out


def _load_ground_truth(
    path: str, video_frames: int, video_fps: float, ppg_fs: float
) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], float, str]:
    """
    Load ground_truth.txt. UBFC can use:
    - 3-row format: row0=PPG, row1=HR (BPM), row2=timestamps (seconds). We use (timestamps, HR) and sync by time.
    - 1-row or 2-row: treat as PPG or sparse HR (existing heuristic).
    Returns (data, sample_rate_or_None, type) with type in ("ppg","hr","hr_ts").
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    rows = [_floats_from_line(ln) for ln in lines]
    # Flatten if one long row split across lines
    all_vals = []
    for r in rows:
        all_vals.extend(r)
    if not all_vals:
        raise ValueError(f"No numeric values in {path}")

    # UBFC 3-row: line0=PPG, line1=HR, line2=timestamps
    if len(rows) >= 3:
        a, b, c = len(rows[0]), len(rows[1]), len(rows[2])
        if a == b == c and c > 10:
            ts = np.array(rows[2], dtype=np.float64)
            hr = np.array(rows[1], dtype=np.float64)
            if np.all(ts >= 0) and np.all(ts <= 600) and np.all(hr >= 30) and np.all(hr <= 200):
                return (ts, hr), 0.0, "hr_ts"

    # Single array: PPG or sparse HR
    arr = np.array(all_vals, dtype=np.float64)
    n = len(arr)
    duration = video_frames / float(video_fps) if video_fps > 0 else 1.0
    expected_ppg = int(duration * ppg_fs)
    if n >= max(1, expected_ppg * 0.5):
        return arr, ppg_fs, "ppg"
    return arr, 1.0 if n <= 1 else (n - 1) / duration, "hr"


def _hr_from_ppg(ppg: np.ndarray, fs: float) -> Optional[float]:
    if len(ppg) < 30:
        return None
    nyq = fs / 2.0
    low, high = 0.7 / nyq, 4.0 / nyq
    if low >= 1 or high >= 1:
        return None
    b, a = signal.butter(4, [low, high], btype="band")
    try:
        filt = signal.filtfilt(b, a, signal.detrend(ppg.astype(np.float64)))
    except Exception:
        return None
    fft = np.abs(np.fft.rfft(filt))
    freqs = np.fft.rfftfreq(len(filt), 1.0 / fs)
    m = (freqs >= 0.7) & (freqs <= 4.0)
    if not np.any(m):
        return None
    f0 = freqs[m][np.argmax(fft[m])]
    bpm = f0 * 60.0
    return float(bpm) if HR_BPM_MIN <= bpm <= HR_BPM_MAX else None


def _mean_hr_for_window(
    t_start: float, t_end: float,
    gt: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], gt_fs: float, gt_type: str
) -> Optional[float]:
    if gt_type == "hr_ts":
        ts, hr = gt
        mask = (ts >= t_start) & (ts <= t_end)
        if np.any(mask):
            return float(np.clip(np.mean(hr[mask]), HR_BPM_MIN, HR_BPM_MAX))
        # Interpolate at center
        tc = (t_start + t_end) / 2
        if tc <= ts[0]:
            return float(np.clip(hr[0], HR_BPM_MIN, HR_BPM_MAX))
        if tc >= ts[-1]:
            return float(np.clip(hr[-1], HR_BPM_MIN, HR_BPM_MAX))
        return float(np.clip(np.interp(tc, ts, hr), HR_BPM_MIN, HR_BPM_MAX))

    if gt_type == "ppg":
        i0 = max(0, int(t_start * gt_fs))
        i1 = min(len(gt), int(t_end * gt_fs) + 1)
        if i1 - i0 < 30:
            return None
        return _hr_from_ppg(gt[i0:i1], gt_fs)

    # hr: sparse
    n = len(gt)
    if n == 0:
        return None
    duration = (n - 1) / gt_fs if n > 1 and gt_fs > 0 else 1.0
    times = np.linspace(0, duration, n) if n > 1 else np.array([0.0])
    tc = (t_start + t_end) / 2
    if tc <= times[0]:
        return float(np.clip(gt[0], HR_BPM_MIN, HR_BPM_MAX))
    if tc >= times[-1]:
        return float(np.clip(gt[-1], HR_BPM_MIN, HR_BPM_MAX))
    return float(np.clip(np.interp(tc, times, gt), HR_BPM_MIN, HR_BPM_MAX))


class UBFCDataGenerator(keras.utils.Sequence):
    """
    Keras Sequence for UBFC-rPPG. Open–process–close per batch; does NOT load videos into RAM.
    __init__: subject_folders = list of paths to subject dirs (each with vid.avi and ground_truth.txt).
    __getitem__: opens vid.avi only for the needed 30-frame window, face-crop, resize 128x128,
                 normalize, sync with ground_truth (HR+timestamps), returns (X, y).
    X: (batch, 30, 128, 128, 3) float32. y: (batch, 1) float32 BPM.
    """

    # Label for static (zero-motion) clips used in contrastive augmentation
    STATIC_HR_LABEL = 0.0

    def __init__(
        self,
        subject_folders: List[str],
        batch_size: int = 16,
        clip_len: int = CLIP_LEN,
        face_size: Tuple[int, int] = FACE_SIZE,
        video_fps: float = UBFC_DEFAULT_FPS,
        ppg_fs: float = UBFC_PPG_FS,
        shuffle: bool = True,
        normalize: str = "divide",
        use_face_detection: bool = True,
        static_ratio: float = 0.0,
    ):
        self.subject_folders = list(subject_folders)
        self.batch_size = batch_size
        self.clip_len = clip_len
        self.face_size = face_size
        self.video_fps = video_fps
        self.ppg_fs = ppg_fs
        self.shuffle = shuffle
        self.normalize = normalize
        self.static_ratio = max(0.0, min(1.0, float(static_ratio)))
        self.face_landmarker = _build_face_detector() if (use_face_detection and _MEDIAPIPE_AVAILABLE) else None

        self._video_info: dict = {}
        self._index: List[Tuple[str, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        self._video_info.clear()
        self._index.clear()
        for subj in self.subject_folders:
            vid = _find_video(subj)
            gt_path = _find_ground_truth(subj)
            if vid is None or gt_path is None:
                continue
            cap = cv2.VideoCapture(vid)
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or self.video_fps
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            try:
                gt_data, gt_fs, gt_type = _load_ground_truth(gt_path, total, fps, self.ppg_fs)
            except Exception:
                continue
            self._video_info[subj] = (vid, total, float(fps), gt_data, gt_fs, gt_type)
            for start in range(0, total - self.clip_len + 1, self.clip_len):
                t_start = start / fps
                t_end = (start + self.clip_len) / fps
                if _mean_hr_for_window(t_start, t_end, gt_data, gt_fs, gt_type) is not None:
                    self._index.append((subj, start))

    def _n_real_n_static(self) -> Tuple[int, int]:
        n_static = int(self.batch_size * self.static_ratio)
        n_real = self.batch_size - n_static
        return (n_real, n_static)

    def __len__(self) -> int:
        n_real, _ = self._n_real_n_static()
        step = max(1, n_real)
        return int(np.ceil(len(self._index) / step))

    def on_epoch_end(self) -> None:
        if self.shuffle and self._index:
            np.random.shuffle(self._index)

    def _make_static_clip(self, subj: str, start_frame: int) -> Optional[np.ndarray]:
        """Create a 30-frame clip from a single repeated frame (zero motion). Returns (clip_len, H, W, 3) or None."""
        if subj not in self._video_info:
            return None
        path, _fc, _fps, _gt, _gfs, _gty = self._video_info[subj]
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        bbox = _detect_face_bbox(frame, self.face_landmarker) if self.face_landmarker else None
        if bbox is not None:
            x, y, w, h = bbox
            frame = frame[y : y + h, x : x + w]
        f = cv2.resize(frame, (self.face_size[1], self.face_size[0]), interpolation=cv2.INTER_LINEAR)
        f = f.astype(np.float32)
        if self.normalize == "divide":
            f = f / 255.0
        # Repeat single frame clip_len times -> (clip_len, H, W, 3)
        stack = [f] * self.clip_len
        return np.stack(stack, axis=0)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        n_real, n_static = self._n_real_n_static()
        step = max(1, n_real)
        start = idx * step
        batch_indices = self._index[start : start + n_real]
        X_list, y_list = [], []

        for subj, start_frame in batch_indices:
            path, _fc, fps, gt_data, gt_fs, gt_type = self._video_info[subj]
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
            frames = []
            for _ in range(self.clip_len):
                ret, f = cap.read()
                if not ret or f is None:
                    break
                frames.append(f)
            cap.release()

            if len(frames) != self.clip_len:
                continue

            bbox = _detect_face_bbox(frames[0], self.face_landmarker) if self.face_landmarker else None
            stack = []
            for f in frames:
                if bbox is not None:
                    x, y, w, h = bbox
                    f = f[y : y + h, x : x + w]
                f = cv2.resize(f, (self.face_size[1], self.face_size[0]), interpolation=cv2.INTER_LINEAR)
                f = f.astype(np.float32)
                if self.normalize == "divide":
                    f = f / 255.0
                stack.append(f)
            x = np.stack(stack, axis=0)

            t_start = start_frame / fps
            t_end = (start_frame + self.clip_len) / fps
            hr = _mean_hr_for_window(t_start, t_end, gt_data, gt_fs, gt_type)
            if hr is None:
                continue

            X_list.append(x)
            y_list.append(hr)

        # Inject static (zero-motion) samples: 1 frame repeated clip_len times, label STATIC_HR_LABEL (0.0)
        for _ in range(n_static):
            if not self._index:
                break
            i = int(np.random.randint(0, len(self._index)))
            subj, start_frame = self._index[i]
            static_clip = self._make_static_clip(subj, start_frame)
            if static_clip is not None:
                X_list.append(static_clip)
                y_list.append(self.STATIC_HR_LABEL)

        if not X_list:
            X = np.zeros((1, self.clip_len, self.face_size[0], self.face_size[1], 3), dtype=np.float32)
            y = np.zeros((1, 1), dtype=np.float32)
            return (X, y)

        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.float32).reshape(-1, 1)

        # Shuffle batch so real and static are mixed
        perm = np.random.permutation(X.shape[0])
        X = X[perm]
        y = y[perm]
        return (X, y)
