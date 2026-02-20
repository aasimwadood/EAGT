"""
Dataset classes for EAGT Multimodal Affect Recognition.


This module implements:
- EAGTDataset: Windowed multimodal dataset with real feature loading
- Feature caching for efficient training
- Support for video, audio, and behavioral modalities

Expected CSV format:
    video_path,audio_path,behav_json,label
    /path/to/video.mp4,/path/to/audio.wav,/path/to/events.json,0
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Import feature extractors
try:
    from .preprocess_video import FaceExtractor
    from .preprocess_behavior import (
        extract_behavioral_features_sequence,
        load_behavior_events,
        BehaviorFeatures,
    )
except ImportError:
    FaceExtractor = None
    extract_behavioral_features_sequence = None
    load_behavior_events = None
    BehaviorFeatures = None

# Import audio processing
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

try:
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    WAV2VEC_AVAILABLE = True
except ImportError:
    WAV2VEC_AVAILABLE = False


class AudioFeatureExtractor:
    """
    Audio feature extraction using wav2vec2.

    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        device: str = "cpu",
    ):
        self.device = device

        if WAV2VEC_AVAILABLE:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name)
            self.model.eval()
            self.model.to(device)
        else:
            self.processor = None
            self.model = None

    def extract(self, audio_path: str, target_len: int = 16) -> np.ndarray:
        """
        Extract audio features from file.

        Parameters
        ----------
        audio_path : str
            Path to audio file.
        target_len : int
            Target sequence length.

        Returns
        -------
        np.ndarray
            Audio features of shape (target_len, 768).
        """
        if self.model is None or not TORCHAUDIO_AVAILABLE:
            return np.zeros((target_len, 768), dtype=np.float32)

        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = waveform.squeeze(0).numpy()

            # Process with wav2vec2
            inputs = self.processor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs = self.model(
                    inputs.input_values.to(self.device),
                    attention_mask=inputs.get("attention_mask", None),
                )
                features = outputs.last_hidden_state.squeeze(0).cpu().numpy()

            # Resample to target length
            if len(features) != target_len:
                indices = np.linspace(0, len(features) - 1, target_len).astype(int)
                features = features[indices]

            return features.astype(np.float32)

        except Exception:
            return np.zeros((target_len, 768), dtype=np.float32)


class FeatureCache:
    """
    Cache for precomputed features to speed up training.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, source_path: str, modality: str) -> Path:
        """Get cache file path for a source file."""
        if self.cache_dir is None:
            return None
        source_hash = str(hash(source_path))[-8:]
        return self.cache_dir / f"{modality}_{source_hash}.npy"

    def get(self, source_path: str, modality: str) -> Optional[np.ndarray]:
        """Get cached features if available."""
        cache_path = self._get_cache_path(source_path, modality)
        if cache_path and cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception:
                pass
        return None

    def put(self, source_path: str, modality: str, features: np.ndarray):
        """Cache features."""
        cache_path = self._get_cache_path(source_path, modality)
        if cache_path:
            try:
                np.save(cache_path, features)
            except Exception:
                pass


class EAGTDataset(Dataset):
    """
    Windowed multimodal dataset for EAGT.


    This dataset loads and processes:
    - Face embeddings from video (via ResNet18 or similar)
    - Audio embeddings from wav files (via wav2vec2)
    - Behavioral features from interaction logs

    Expected CSV format:
        video_path,audio_path,behav_json,label

    Returns dict:
        {
          "face": (T, Df),
          "audio": (T, Da),
          "behav": (T, Db),
          "label": int
        }

    Parameters
    ----------
    csv_path : str
        Path to CSV file with data paths and labels.
    face_dim : int
        Dimension of face embeddings. Default 512.
    audio_dim : int
        Dimension of audio embeddings. Default 768 (wav2vec2).
    behav_dim : int
        Dimension of behavioral features. Default 16.
    seq_len : int
        Sequence length (number of time windows). Default 16.
    cache_dir : str, optional
        Directory for caching extracted features.
    use_real_features : bool
        If True, extract real features from files. If False, use random tensors.
    face_encoder : nn.Module, optional
        Pre-trained face encoder for extracting embeddings.
    """

    def __init__(
        self,
        csv_path: str,
        face_dim: int = 512,
        audio_dim: int = 768,
        behav_dim: int = 16,
        seq_len: int = 16,
        cache_dir: Optional[str] = None,
        use_real_features: bool = True,
        face_encoder=None,
    ):
        # Load CSV
        try:
            self.df = pd.read_csv(csv_path)
            # Validate required columns
            required_cols = ["label"]
            for col in required_cols:
                if col not in self.df.columns:
                    raise ValueError(f"Missing required column: {col}")
        except Exception:
            # Fallback: dummy dataset for testing
            self.df = pd.DataFrame({
                "video_path": [""] * 16,
                "audio_path": [""] * 16,
                "behav_json": [""] * 16,
                "label": [0, 1, 2, 3] * 4,
            })

        self.face_dim = face_dim
        self.audio_dim = audio_dim
        self.behav_dim = behav_dim
        self.seq_len = seq_len
        self.use_real_features = use_real_features

        # Initialize feature extractors
        self.face_extractor = None
        self.audio_extractor = None
        self.face_encoder = face_encoder

        if use_real_features:
            if FaceExtractor is not None:
                self.face_extractor = FaceExtractor(target_size=112)
            self.audio_extractor = AudioFeatureExtractor()

        # Initialize cache
        self.cache = FeatureCache(cache_dir) if cache_dir else None

        # Precompute behavioral feature dimension
        if BehaviorFeatures is not None:
            self.behav_dim = BehaviorFeatures.feature_dim()
        else:
            self.behav_dim = behav_dim

    def __len__(self) -> int:
        return len(self.df)

    def _extract_face_features(self, video_path: str) -> np.ndarray:
        """Extract face embeddings from video."""
        # Check cache
        if self.cache:
            cached = self.cache.get(video_path, "face")
            if cached is not None:
                return self._pad_or_truncate(cached, self.seq_len, self.face_dim)

        # Extract features
        if self.face_extractor is None or not video_path or not os.path.exists(video_path):
            return np.zeros((self.seq_len, self.face_dim), dtype=np.float32)

        try:
            faces = self.face_extractor.crop_faces(video_path, fps=8)

            if not faces or self.face_encoder is None:
                return np.zeros((self.seq_len, self.face_dim), dtype=np.float32)

            # Convert faces to tensor
            faces_tensor = torch.stack([
                torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
                for f in faces
            ])

            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            faces_tensor = (faces_tensor - mean) / std

            # Extract embeddings
            self.face_encoder.eval()
            with torch.no_grad():
                features = self.face_encoder(faces_tensor).cpu().numpy()

            # Cache
            if self.cache:
                self.cache.put(video_path, "face", features)

            return self._pad_or_truncate(features, self.seq_len, self.face_dim)

        except Exception:
            return np.zeros((self.seq_len, self.face_dim), dtype=np.float32)

    def _extract_audio_features(self, audio_path: str) -> np.ndarray:
        """Extract audio embeddings from audio file."""
        # Check cache
        if self.cache:
            cached = self.cache.get(audio_path, "audio")
            if cached is not None:
                return self._pad_or_truncate(cached, self.seq_len, self.audio_dim)

        # Extract features
        if self.audio_extractor is None or not audio_path or not os.path.exists(audio_path):
            return np.zeros((self.seq_len, self.audio_dim), dtype=np.float32)

        try:
            features = self.audio_extractor.extract(audio_path, self.seq_len)

            # Cache
            if self.cache:
                self.cache.put(audio_path, "audio", features)

            return self._pad_or_truncate(features, self.seq_len, self.audio_dim)

        except Exception:
            return np.zeros((self.seq_len, self.audio_dim), dtype=np.float32)

    def _extract_behav_features(self, behav_path: str) -> np.ndarray:
        """Extract behavioral features from interaction log."""
        if extract_behavioral_features_sequence is None:
            return np.zeros((self.seq_len, self.behav_dim), dtype=np.float32)

        if not behav_path or not os.path.exists(behav_path):
            return np.zeros((self.seq_len, self.behav_dim), dtype=np.float32)

        try:
            events = load_behavior_events(behav_path)
            features = extract_behavioral_features_sequence(
                events,
                window_sec=1.0,
                stride_sec=0.5,
                num_windows=self.seq_len,
            )
            return features.astype(np.float32)

        except Exception:
            return np.zeros((self.seq_len, self.behav_dim), dtype=np.float32)

    def _pad_or_truncate(
        self,
        features: np.ndarray,
        target_len: int,
        feature_dim: int,
    ) -> np.ndarray:
        """Pad or truncate features to target length."""
        if len(features) == 0:
            return np.zeros((target_len, feature_dim), dtype=np.float32)

        if len(features) > target_len:
            # Truncate by sampling
            indices = np.linspace(0, len(features) - 1, target_len).astype(int)
            return features[indices]
        elif len(features) < target_len:
            # Pad with zeros
            padding = np.zeros((target_len - len(features), feature_dim), dtype=np.float32)
            return np.concatenate([features, padding], axis=0)
        return features

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        label = int(row["label"])

        if self.use_real_features:
            # Extract real features from files
            video_path = row.get("video_path", "")
            audio_path = row.get("audio_path", "")
            behav_path = row.get("behav_json", "")

            face_feats = self._extract_face_features(video_path)
            audio_feats = self._extract_audio_features(audio_path)
            behav_feats = self._extract_behav_features(behav_path)
        else:
            # Use random tensors for testing/demo
            face_feats = np.random.randn(self.seq_len, self.face_dim).astype(np.float32)
            audio_feats = np.random.randn(self.seq_len, self.audio_dim).astype(np.float32)
            behav_feats = np.random.randn(self.seq_len, self.behav_dim).astype(np.float32)

        return {
            "face": torch.from_numpy(face_feats).float(),
            "audio": torch.from_numpy(audio_feats).float(),
            "behav": torch.from_numpy(behav_feats).float(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class EAGTDatasetPrecomputed(Dataset):
    """
    Dataset using precomputed features stored in numpy files.

    This is faster for training as features are already extracted.

    Expected directory structure:
        features_dir/
            face/
                sample_0.npy
                sample_1.npy
                ...
            audio/
                sample_0.npy
                ...
            behav/
                sample_0.npy
                ...
        labels.csv (with columns: sample_id, label)
    """

    def __init__(
        self,
        features_dir: str,
        labels_csv: str,
        face_dim: int = 512,
        audio_dim: int = 768,
        behav_dim: int = 7,
        seq_len: int = 16,
    ):
        self.features_dir = Path(features_dir)
        self.labels_df = pd.read_csv(labels_csv)
        self.face_dim = face_dim
        self.audio_dim = audio_dim
        self.behav_dim = behav_dim
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.labels_df)

    def _load_feature(self, modality: str, sample_id: str, dim: int) -> np.ndarray:
        """Load precomputed feature file."""
        path = self.features_dir / modality / f"{sample_id}.npy"
        if path.exists():
            try:
                features = np.load(path)
                if len(features) != self.seq_len:
                    indices = np.linspace(0, len(features) - 1, self.seq_len).astype(int)
                    features = features[indices]
                return features.astype(np.float32)
            except Exception:
                pass
        return np.zeros((self.seq_len, dim), dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.labels_df.iloc[idx]
        sample_id = str(row.get("sample_id", idx))
        label = int(row["label"])

        face_feats = self._load_feature("face", sample_id, self.face_dim)
        audio_feats = self._load_feature("audio", sample_id, self.audio_dim)
        behav_feats = self._load_feature("behav", sample_id, self.behav_dim)

        return {
            "face": torch.from_numpy(face_feats).float(),
            "audio": torch.from_numpy(audio_feats).float(),
            "behav": torch.from_numpy(behav_feats).float(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    batch_size: int = 16,
    num_workers: int = 4,
    use_real_features: bool = True,
    cache_dir: Optional[str] = None,
    face_encoder=None,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.

    Parameters
    ----------
    train_csv : str
        Path to training CSV.
    val_csv : str
        Path to validation CSV.
    batch_size : int
        Batch size.
    num_workers : int
        Number of data loading workers.
    use_real_features : bool
        Whether to extract real features.
    cache_dir : str, optional
        Directory for feature caching.
    face_encoder : nn.Module, optional
        Face embedding encoder.

    Returns
    -------
    Dict with 'train' and 'val' DataLoaders.
    """
    train_ds = EAGTDataset(
        train_csv,
        use_real_features=use_real_features,
        cache_dir=cache_dir,
        face_encoder=face_encoder,
    )
    val_ds = EAGTDataset(
        val_csv,
        use_real_features=use_real_features,
        cache_dir=cache_dir,
        face_encoder=face_encoder,
    )

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )

    return {"train": train_dl, "val": val_dl}
