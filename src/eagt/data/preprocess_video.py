"""
Video Preprocessing with Face Detection, CLAHE, and FACS.


This module implements:
- Face detection using MediaPipe or OpenCV cascades
- Face alignment and cropping
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for illumination normalization
- Facial Action Coding System (FACS) feature extraction
- Facial landmark extraction

MediaPipe/OpenFace face detection, alignment, CLAHE
illumination normalization, Facial Action Coding System features.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
from dataclasses import dataclass

# Try to import MediaPipe (optional dependency)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None


@dataclass
class FaceDetectionResult:
    """Result from face detection."""
    face_crop: np.ndarray  # RGB face crop
    landmarks: Optional[np.ndarray] = None  # Facial landmarks (68 or 478 points)
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    confidence: float = 1.0
    action_units: Optional[Dict[str, float]] = None  # FACS Action Units


class CLAHEProcessor:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) processor.


    CLAHE improves face recognition under varying illumination conditions
    by enhancing local contrast while limiting noise amplification.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ):
        """
        Parameters
        ----------
        clip_limit : float
            Threshold for contrast limiting. Default 2.0.
        tile_grid_size : Tuple[int, int]
            Size of grid for histogram equalization. Default (8, 8).
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to an image.

        Parameters
        ----------
        image : np.ndarray
            Input image in RGB or BGR format.

        Returns
        -------
        np.ndarray
            CLAHE-enhanced image.
        """
        if image.ndim == 3:
            # Convert to LAB color space
            if image.shape[2] == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                # Apply CLAHE to L channel
                lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
                # Convert back to RGB
                return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        elif image.ndim == 2:
            # Grayscale image
            return self.clahe.apply(image)

        return image


class FaceDetectorMediaPipe:
    """
    Face detection using MediaPipe Face Detection and Face Mesh.


    MediaPipe provides robust face detection and 478 facial landmarks
    for detailed face analysis.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe not available. Install with: pip install mediapipe"
            )

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
        )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, image: np.ndarray) -> Optional[FaceDetectionResult]:
        """
        Detect face and extract landmarks.

        Parameters
        ----------
        image : np.ndarray
            RGB image.

        Returns
        -------
        FaceDetectionResult or None if no face detected.
        """
        h, w = image.shape[:2]

        # Run face mesh for landmarks
        results = self.face_mesh.process(image)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]

        # Extract landmark coordinates
        landmark_coords = np.array([
            [lm.x * w, lm.y * h, lm.z * w]
            for lm in landmarks.landmark
        ])

        # Compute bounding box from landmarks
        x_coords = landmark_coords[:, 0]
        y_coords = landmark_coords[:, 1]

        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())

        # Add padding
        pad = int(0.1 * max(x_max - x_min, y_max - y_min))
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)

        # Crop face
        face_crop = image[y_min:y_max, x_min:x_max].copy()

        return FaceDetectionResult(
            face_crop=face_crop,
            landmarks=landmark_coords,
            bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
            confidence=1.0,
        )

    def close(self):
        """Release resources."""
        self.face_detection.close()
        self.face_mesh.close()


class FaceDetectorCascade:
    """
    Fallback face detection using OpenCV Haar Cascades.

    Used when MediaPipe is not available.
    """

    def __init__(self):
        # Load Haar cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, image: np.ndarray) -> Optional[FaceDetectionResult]:
        """Detect face using Haar cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        if len(faces) == 0:
            return None

        # Take largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Add padding
        pad = int(0.1 * max(w, h))
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(image.shape[1] - x, w + 2 * pad)
        h = min(image.shape[0] - y, h + 2 * pad)

        face_crop = image[y:y+h, x:x+w].copy()

        return FaceDetectionResult(
            face_crop=face_crop,
            landmarks=None,
            bbox=(x, y, w, h),
            confidence=0.9,
        )

    def close(self):
        pass


class FACSExtractor:
    """
    Facial Action Coding System (FACS) feature extraction.


    FACS describes facial expressions in terms of Action Units (AUs),
    which represent individual muscle movements. Key AUs for affect:

    - AU1: Inner brow raiser (sadness, fear)
    - AU2: Outer brow raiser (surprise)
    - AU4: Brow lowerer (anger, confusion)
    - AU5: Upper lid raiser (surprise, fear)
    - AU6: Cheek raiser (happiness)
    - AU7: Lid tightener (anger)
    - AU9: Nose wrinkler (disgust)
    - AU10: Upper lip raiser (disgust)
    - AU12: Lip corner puller (happiness)
    - AU14: Dimpler
    - AU15: Lip corner depressor (sadness)
    - AU17: Chin raiser
    - AU20: Lip stretcher (fear)
    - AU23: Lip tightener (anger)
    - AU25: Lips part
    - AU26: Jaw drop (surprise)
    - AU45: Blink

    This implementation uses geometric features from landmarks as proxies
    for AU activations. For production, consider using OpenFace or similar.
    """

    # MediaPipe landmark indices for key facial points
    # These are approximate mappings to the 478-point mesh
    LANDMARK_INDICES = {
        "left_eyebrow_inner": 107,
        "left_eyebrow_outer": 70,
        "right_eyebrow_inner": 336,
        "right_eyebrow_outer": 300,
        "left_eye_upper": 159,
        "left_eye_lower": 145,
        "right_eye_upper": 386,
        "right_eye_lower": 374,
        "nose_tip": 1,
        "left_mouth_corner": 61,
        "right_mouth_corner": 291,
        "upper_lip": 13,
        "lower_lip": 14,
        "chin": 152,
        "left_cheek": 123,
        "right_cheek": 352,
    }

    def __init__(self):
        self.au_names = [
            "AU1", "AU2", "AU4", "AU5", "AU6", "AU7",
            "AU9", "AU10", "AU12", "AU14", "AU15",
            "AU17", "AU20", "AU23", "AU25", "AU26", "AU45",
        ]

    def extract(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract Action Unit activations from landmarks.

        Parameters
        ----------
        landmarks : np.ndarray
            Facial landmarks of shape (N, 3) where N is 478 for MediaPipe.

        Returns
        -------
        Dict[str, float]
            Action Unit activations in [0, 1] range.
        """
        if landmarks is None or len(landmarks) < 100:
            # Return neutral if no landmarks
            return {au: 0.0 for au in self.au_names}

        aus = {}

        try:
            # Get key landmark positions
            idx = self.LANDMARK_INDICES

            # Compute reference distances for normalization
            face_height = np.linalg.norm(
                landmarks[idx["nose_tip"]] - landmarks[idx["chin"]]
            )
            if face_height < 1:
                face_height = 1

            # AU1/AU2: Brow raise (inner/outer)
            left_brow_height = landmarks[idx["left_eyebrow_inner"]][1]
            left_eye_height = landmarks[idx["left_eye_upper"]][1]
            brow_eye_dist = abs(left_brow_height - left_eye_height) / face_height
            aus["AU1"] = min(1.0, brow_eye_dist * 5)
            aus["AU2"] = min(1.0, brow_eye_dist * 5)

            # AU4: Brow lowerer
            brow_dist = np.linalg.norm(
                landmarks[idx["left_eyebrow_inner"]] -
                landmarks[idx["right_eyebrow_inner"]]
            )
            aus["AU4"] = max(0, 1.0 - brow_dist / face_height * 3)

            # AU5: Upper lid raiser
            left_eye_openness = np.linalg.norm(
                landmarks[idx["left_eye_upper"]] -
                landmarks[idx["left_eye_lower"]]
            )
            aus["AU5"] = min(1.0, left_eye_openness / face_height * 10)

            # AU6: Cheek raiser
            cheek_raise = landmarks[idx["left_cheek"]][1] - landmarks[idx["left_eye_lower"]][1]
            aus["AU6"] = max(0, min(1.0, -cheek_raise / face_height * 5))

            # AU7: Lid tightener
            aus["AU7"] = max(0, 1.0 - aus["AU5"])

            # AU9/AU10: Nose wrinkler / Upper lip raiser
            nose_lip_dist = np.linalg.norm(
                landmarks[idx["nose_tip"]] - landmarks[idx["upper_lip"]]
            )
            aus["AU9"] = max(0, 1.0 - nose_lip_dist / face_height * 5)
            aus["AU10"] = aus["AU9"]

            # AU12: Lip corner puller (smile)
            mouth_width = np.linalg.norm(
                landmarks[idx["left_mouth_corner"]] -
                landmarks[idx["right_mouth_corner"]]
            )
            mouth_height = np.linalg.norm(
                landmarks[idx["upper_lip"]] - landmarks[idx["lower_lip"]]
            )
            smile_ratio = mouth_width / (mouth_height + 1e-6)
            aus["AU12"] = min(1.0, max(0, (smile_ratio - 2) / 3))

            # AU14: Dimpler (approximated)
            aus["AU14"] = aus["AU12"] * 0.5

            # AU15: Lip corner depressor
            left_corner_y = landmarks[idx["left_mouth_corner"]][1]
            center_lip_y = landmarks[idx["upper_lip"]][1]
            corner_drop = left_corner_y - center_lip_y
            aus["AU15"] = max(0, min(1.0, corner_drop / face_height * 10))

            # AU17: Chin raiser
            chin_raise = landmarks[idx["lower_lip"]][1] - landmarks[idx["chin"]][1]
            aus["AU17"] = max(0, min(1.0, 1.0 - chin_raise / face_height * 5))

            # AU20: Lip stretcher
            aus["AU20"] = min(1.0, mouth_width / face_height * 2)

            # AU23: Lip tightener
            aus["AU23"] = max(0, 1.0 - mouth_height / face_height * 10)

            # AU25: Lips part
            aus["AU25"] = min(1.0, mouth_height / face_height * 15)

            # AU26: Jaw drop
            jaw_drop = landmarks[idx["chin"]][1] - landmarks[idx["nose_tip"]][1]
            aus["AU26"] = min(1.0, max(0, jaw_drop / face_height - 0.3) * 3)

            # AU45: Blink (approximated from eye openness)
            aus["AU45"] = max(0, 1.0 - aus["AU5"])

        except (IndexError, KeyError):
            # Return neutral on error
            return {au: 0.0 for au in self.au_names}

        return aus

    def to_vector(self, aus: Dict[str, float]) -> np.ndarray:
        """Convert AU dict to feature vector."""
        return np.array([aus.get(au, 0.0) for au in self.au_names], dtype=np.float32)

    @property
    def feature_dim(self) -> int:
        return len(self.au_names)


class FaceExtractor:
    """
    Complete face extraction pipeline with detection, CLAHE, and FACS.


    This class provides a production-ready face extraction pipeline:
    1. Face detection (MediaPipe or Haar cascade fallback)
    2. Face alignment and cropping
    3. CLAHE illumination normalization
    4. FACS Action Unit extraction

    Methods
    -------
    crop_faces(video_path: str, fps: int = 8) -> List[np.ndarray]
        Returns a list of RGB face crops (H=W=target_size) sampled at ~fps.

    extract_features(video_path: str, fps: int = 8) -> Dict
        Returns dict with 'faces', 'landmarks', 'action_units'.
    """

    def __init__(
        self,
        target_size: int = 112,
        use_clahe: bool = True,
        use_mediapipe: bool = True,
        extract_facs: bool = True,
    ):
        """
        Parameters
        ----------
        target_size : int
            Output face crop size (square).
        use_clahe : bool
            Whether to apply CLAHE preprocessing.
        use_mediapipe : bool
            Whether to use MediaPipe (falls back to Haar if unavailable).
        extract_facs : bool
            Whether to extract FACS Action Units.
        """
        self.size = target_size
        self.use_clahe = use_clahe
        self.extract_facs = extract_facs

        # Initialize CLAHE
        self.clahe = CLAHEProcessor() if use_clahe else None

        # Initialize face detector
        if use_mediapipe and MEDIAPIPE_AVAILABLE:
            self.detector = FaceDetectorMediaPipe()
        else:
            self.detector = FaceDetectorCascade()

        # Initialize FACS extractor
        self.facs = FACSExtractor() if extract_facs else None

    def _process_frame(self, frame_rgb: np.ndarray) -> Optional[FaceDetectionResult]:
        """Process a single frame."""
        # Detect face
        result = self.detector.detect(frame_rgb)

        if result is None:
            return None

        # Apply CLAHE
        if self.clahe is not None:
            result.face_crop = self.clahe.apply(result.face_crop)

        # Resize to target size
        result.face_crop = cv2.resize(
            result.face_crop,
            (self.size, self.size),
            interpolation=cv2.INTER_AREA,
        )

        # Extract FACS
        if self.facs is not None and result.landmarks is not None:
            result.action_units = self.facs.extract(result.landmarks)

        return result

    def crop_faces(self, video_path: str, fps: int = 8) -> List[np.ndarray]:
        """
        Extract face crops from video.

        Parameters
        ----------
        video_path : str
            Path to video file.
        fps : int
            Target frames per second for sampling.

        Returns
        -------
        List[np.ndarray]
            List of RGB face crops of shape (target_size, target_size, 3).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frames = []
        in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, int(round(in_fps / max(1, fps))))

        i = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if i % stride == 0:
                frame_rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB
                result = self._process_frame(frame_rgb)

                if result is not None:
                    frames.append(result.face_crop)
                else:
                    # Fallback: center crop if no face detected
                    h, w = frame_rgb.shape[:2]
                    s = min(h, w)
                    y0 = (h - s) // 2
                    x0 = (w - s) // 2
                    crop = frame_rgb[y0:y0+s, x0:x0+s]
                    crop = cv2.resize(crop, (self.size, self.size))
                    if self.clahe:
                        crop = self.clahe.apply(crop)
                    frames.append(crop)

            i += 1

        cap.release()
        return frames

    def extract_features(
        self,
        video_path: str,
        fps: int = 8,
    ) -> Dict[str, Union[List[np.ndarray], np.ndarray]]:
        """
        Extract complete face features from video.

        Parameters
        ----------
        video_path : str
            Path to video file.
        fps : int
            Target frames per second.

        Returns
        -------
        Dict with keys:
            - 'faces': List of face crops
            - 'landmarks': Array of landmarks (T, N, 3)
            - 'action_units': Array of FACS features (T, 17)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        faces = []
        landmarks_list = []
        aus_list = []

        in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, int(round(in_fps / max(1, fps))))

        i = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if i % stride == 0:
                frame_rgb = frame_bgr[:, :, ::-1]
                result = self._process_frame(frame_rgb)

                if result is not None:
                    faces.append(result.face_crop)
                    if result.landmarks is not None:
                        landmarks_list.append(result.landmarks)
                    if result.action_units is not None:
                        aus_list.append(self.facs.to_vector(result.action_units))
                else:
                    # Fallback
                    h, w = frame_rgb.shape[:2]
                    s = min(h, w)
                    y0, x0 = (h - s) // 2, (w - s) // 2
                    crop = cv2.resize(frame_rgb[y0:y0+s, x0:x0+s], (self.size, self.size))
                    if self.clahe:
                        crop = self.clahe.apply(crop)
                    faces.append(crop)
                    # Neutral FACS for missing face
                    if self.facs:
                        aus_list.append(np.zeros(self.facs.feature_dim, dtype=np.float32))

            i += 1

        cap.release()

        result = {"faces": faces}

        if landmarks_list:
            result["landmarks"] = np.stack(landmarks_list, axis=0)
        else:
            result["landmarks"] = None

        if aus_list:
            result["action_units"] = np.stack(aus_list, axis=0)
        else:
            result["action_units"] = None

        return result

    def close(self):
        """Release resources."""
        if hasattr(self.detector, 'close'):
            self.detector.close()


def extract_face_embeddings_batch(
    video_paths: List[str],
    encoder,
    face_extractor: Optional[FaceExtractor] = None,
    fps: int = 8,
    batch_size: int = 16,
) -> List[np.ndarray]:
    """
    Extract face embeddings for multiple videos.

    Parameters
    ----------
    video_paths : List[str]
        List of video file paths.
    encoder : nn.Module
        Face embedding model (e.g., ResNet18).
    face_extractor : FaceExtractor, optional
        Face extraction pipeline.
    fps : int
        Target FPS for face sampling.
    batch_size : int
        Batch size for encoder.

    Returns
    -------
    List[np.ndarray]
        List of embedding arrays, one per video.
    """
    import torch

    if face_extractor is None:
        face_extractor = FaceExtractor()

    all_embeddings = []

    for video_path in video_paths:
        try:
            faces = face_extractor.crop_faces(video_path, fps=fps)
        except Exception:
            faces = []

        if not faces:
            all_embeddings.append(np.zeros((1, 512), dtype=np.float32))
            continue

        # Convert to tensor
        faces_tensor = torch.stack([
            torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
            for f in faces
        ])

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        faces_tensor = (faces_tensor - mean) / std

        # Extract embeddings in batches
        embeddings = []
        encoder.eval()
        with torch.no_grad():
            for i in range(0, len(faces_tensor), batch_size):
                batch = faces_tensor[i:i+batch_size]
                emb = encoder(batch)
                embeddings.append(emb.cpu().numpy())

        all_embeddings.append(np.concatenate(embeddings, axis=0))

    return all_embeddings
