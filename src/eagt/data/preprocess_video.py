import cv2
from typing import List
import numpy as np

class FaceExtractor:
    """
    Minimal face cropper placeholder.

    For production:
      - Replace center-crop with a real detector (MTCNN via facenet-pytorch,
        or MediaPipe FaceDetection/FaceMesh).
      - Cache detected face boxes and embeddings to disk to speed up training.

    Methods
    -------
    crop_faces(video_path: str, fps: int = 8) -> List[np.ndarray]
        Returns a list of RGB face crops (H=W=target_size) sampled at ~fps.
    """
    def __init__(self, target_size: int = 112):
        self.size = target_size

    def crop_faces(self, video_path: str, fps: int = 8) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frames = []
        # approximate downsampling to desired fps
        in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, int(round(in_fps / max(1, fps))))

        i = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if i % stride == 0:
                h, w = frame_bgr.shape[:2]
                s = min(h, w)
                y0 = (h - s) // 2
                x0 = (w - s) // 2
                crop = frame_bgr[y0:y0+s, x0:x0+s]
                crop = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_AREA)
                frames.append(crop[:, :, ::-1])  # BGR -> RGB
            i += 1

        cap.release()
        return frames
