"""
Data Augmentation for Multimodal Affect Recognition.


This module implements augmentation strategies:
- Gaussian noise injection to audio
- Random temporal masking
- Modality dropout for robustness to missing data
- Random cropping/scaling for video frames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    """Configuration for multimodal augmentation."""
    # Audio augmentation
    audio_noise_std: float = 0.01
    audio_time_stretch_range: Tuple[float, float] = (0.9, 1.1)
    audio_pitch_shift_range: Tuple[float, float] = (-2, 2)

    # Video augmentation
    video_brightness_range: Tuple[float, float] = (0.8, 1.2)
    video_contrast_range: Tuple[float, float] = (0.8, 1.2)
    video_random_crop_scale: Tuple[float, float] = (0.8, 1.0)

    # Temporal augmentation
    time_mask_prob: float = 0.1
    time_mask_max_ratio: float = 0.25

    # Modality dropout (for robustness)
    modality_dropout_prob: float = 0.1

    # General
    p: float = 0.5  # Overall probability of applying augmentation


class GaussianNoise(nn.Module):
    """Add Gaussian noise to input tensor."""

    def __init__(self, std: float = 0.01):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


class TemporalMask(nn.Module):
    """
    Randomly mask temporal segments.
    data augmentation
    to simulate real-world noise and occlusion.
    """

    def __init__(self, mask_prob: float = 0.1, max_mask_ratio: float = 0.25):
        super().__init__()
        self.mask_prob = mask_prob
        self.max_mask_ratio = max_mask_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, T, D) or (T, D)

        Returns
        -------
        torch.Tensor
            Masked input with same shape.
        """
        if not self.training:
            return x

        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, T, D = x.shape
        x = x.clone()

        for b in range(B):
            if np.random.random() < self.mask_prob:
                max_mask_len = max(1, int(T * self.max_mask_ratio))
                mask_len = np.random.randint(1, max_mask_len + 1)
                mask_start = np.random.randint(0, max(1, T - mask_len + 1))
                x[b, mask_start:mask_start + mask_len] = 0

        if squeeze:
            x = x.squeeze(0)

        return x


class ModalityDropout(nn.Module):
    """
    Randomly drop entire modalities during training.
    modality dropout
    for robustness to prolonged modality absence.
    """

    def __init__(self, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout_prob = dropout_prob

    def forward(
        self,
        face: torch.Tensor,
        audio: torch.Tensor,
        behav: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly zero out entire modalities.

        Parameters
        ----------
        face : torch.Tensor
            Face features (B, T, Df)
        audio : torch.Tensor
            Audio features (B, T, Da)
        behav : torch.Tensor
            Behavioral features (B, T, Db)

        Returns
        -------
        Tuple of (face, audio, behav) with potential dropouts.
        """
        if not self.training:
            return face, audio, behav

        B = face.shape[0]
        face = face.clone()
        audio = audio.clone()
        behav = behav.clone()

        for b in range(B):
            if np.random.random() < self.dropout_prob:
                modality = np.random.choice(["face", "audio", "behav"])
                if modality == "face":
                    face[b] = 0
                elif modality == "audio":
                    audio[b] = 0
                else:
                    behav[b] = 0

        return face, audio, behav


class SpecAugment(nn.Module):
    """
    SpecAugment-style augmentation for audio features.

    Applies frequency and time masking commonly used in speech recognition.
    """

    def __init__(
        self,
        freq_mask_param: int = 10,
        time_mask_param: int = 20,
        num_freq_masks: int = 1,
        num_time_masks: int = 1,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Audio features of shape (B, T, D) or (T, D)
        """
        if not self.training:
            return x

        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, T, D = x.shape
        x = x.clone()

        for b in range(B):
            # Frequency masking
            for _ in range(self.num_freq_masks):
                f = np.random.randint(0, min(self.freq_mask_param, D))
                f0 = np.random.randint(0, max(1, D - f))
                x[b, :, f0:f0 + f] = 0

            # Time masking
            for _ in range(self.num_time_masks):
                t = np.random.randint(0, min(self.time_mask_param, T))
                t0 = np.random.randint(0, max(1, T - t))
                x[b, t0:t0 + t, :] = 0

        if squeeze:
            x = x.squeeze(0)

        return x


class MultimodalAugmentation(nn.Module):
    """
    Combined augmentation pipeline for multimodal affect recognition.


    This module applies various augmentations to face, audio, and behavioral
    features to improve model robustness and generalization.
    """

    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        audio_noise_std: float = 0.01,
        time_mask_prob: float = 0.1,
        modality_dropout_prob: float = 0.1,
    ):
        super().__init__()

        if config is not None:
            self.config = config
        else:
            self.config = AugmentationConfig(
                audio_noise_std=audio_noise_std,
                time_mask_prob=time_mask_prob,
                modality_dropout_prob=modality_dropout_prob,
            )

        # Audio augmentations
        self.audio_noise = GaussianNoise(self.config.audio_noise_std)
        self.audio_spec_augment = SpecAugment(
            freq_mask_param=10,
            time_mask_param=5,
            num_freq_masks=1,
            num_time_masks=1,
        )

        # Temporal augmentations
        self.temporal_mask = TemporalMask(
            self.config.time_mask_prob,
            self.config.time_mask_max_ratio,
        )

        # Modality dropout
        self.modality_dropout = ModalityDropout(self.config.modality_dropout_prob)

    def forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply augmentations to a batch.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Dictionary with keys "face", "audio", "behav", "label"

        Returns
        -------
        Dict[str, torch.Tensor]
            Augmented batch with same structure.
        """
        face = batch["face"].clone()
        audio = batch["audio"].clone()
        behav = batch["behav"].clone()
        label = batch["label"]

        if self.training:
            # Apply audio noise
            audio = self.audio_noise(audio)

            # Apply SpecAugment to audio
            if np.random.random() < self.config.p:
                audio = self.audio_spec_augment(audio)

            # Apply temporal masking to all modalities
            if np.random.random() < self.config.p:
                face = self.temporal_mask(face)
                audio = self.temporal_mask(audio)

            # Apply modality dropout
            face, audio, behav = self.modality_dropout(face, audio, behav)

        return {
            "face": face,
            "audio": audio,
            "behav": behav,
            "label": label,
        }

    def augment_face(self, face: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to face features only."""
        if self.training:
            face = self.temporal_mask(face)
        return face

    def augment_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to audio features only."""
        if self.training:
            audio = self.audio_noise(audio)
            if np.random.random() < self.config.p:
                audio = self.audio_spec_augment(audio)
            audio = self.temporal_mask(audio)
        return audio

    def augment_behav(self, behav: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to behavioral features only."""
        if self.training:
            # Add small noise to behavioral features
            noise = torch.randn_like(behav) * 0.05
            behav = behav + noise
        return behav


class MixUp(nn.Module):
    """
    MixUp augmentation for multimodal data.

    Creates convex combinations of training examples.
    """

    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp to batch.

        Returns
        -------
        Tuple of (mixed_batch, labels_a, labels_b, lambda)
        """
        face = batch["face"]
        audio = batch["audio"]
        behav = batch["behav"]
        labels = batch["label"]

        B = face.shape[0]

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0

        # Random permutation for mixing
        index = torch.randperm(B, device=face.device)

        # Mix features
        mixed_face = lam * face + (1 - lam) * face[index]
        mixed_audio = lam * audio + (1 - lam) * audio[index]
        mixed_behav = lam * behav + (1 - lam) * behav[index]

        mixed_batch = {
            "face": mixed_face,
            "audio": mixed_audio,
            "behav": mixed_behav,
            "label": labels,  # Original labels
        }

        return mixed_batch, labels, labels[index], lam


class CutMix(nn.Module):
    """
    CutMix augmentation adapted for temporal sequences.

    Replaces a temporal segment with data from another sample.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, float]:
        """
        Apply temporal CutMix to batch.
        """
        face = batch["face"]
        audio = batch["audio"]
        behav = batch["behav"]
        labels = batch["label"]

        B, T, _ = face.shape

        # Sample lambda
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0

        # Compute cut boundaries
        cut_len = int(T * (1 - lam))
        cut_start = np.random.randint(0, max(1, T - cut_len + 1))
        cut_end = cut_start + cut_len

        # Random permutation
        index = torch.randperm(B, device=face.device)

        # Apply cut
        mixed_face = face.clone()
        mixed_audio = audio.clone()
        mixed_behav = behav.clone()

        mixed_face[:, cut_start:cut_end] = face[index, cut_start:cut_end]
        mixed_audio[:, cut_start:cut_end] = audio[index, cut_start:cut_end]
        mixed_behav[:, cut_start:cut_end] = behav[index, cut_start:cut_end]

        # Adjust lambda based on actual cut
        lam = 1 - (cut_end - cut_start) / T

        mixed_batch = {
            "face": mixed_face,
            "audio": mixed_audio,
            "behav": mixed_behav,
            "label": labels,
        }

        return mixed_batch, labels, labels[index], lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Compute loss for MixUp/CutMix augmented data.

    Parameters
    ----------
    criterion : nn.Module
        Loss function (e.g., CrossEntropyLoss)
    pred : torch.Tensor
        Model predictions
    y_a, y_b : torch.Tensor
        Original and mixed labels
    lam : float
        Mixing coefficient

    Returns
    -------
    torch.Tensor
        Combined loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_augmentation_pipeline(
    audio_noise_std: float = 0.01,
    time_mask_prob: float = 0.1,
    modality_dropout_prob: float = 0.1,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
) -> nn.Module:
    """
    Factory function to create augmentation pipeline.

    Parameters
    ----------
    audio_noise_std : float
        Standard deviation for Gaussian noise on audio.
    time_mask_prob : float
        Probability of applying temporal masking.
    modality_dropout_prob : float
        Probability of dropping a modality.
    use_mixup : bool
        Whether to include MixUp augmentation.
    mixup_alpha : float
        Alpha parameter for MixUp Beta distribution.

    Returns
    -------
    nn.Module
        Augmentation module.
    """
    config = AugmentationConfig(
        audio_noise_std=audio_noise_std,
        time_mask_prob=time_mask_prob,
        modality_dropout_prob=modality_dropout_prob,
    )

    return MultimodalAugmentation(config)
