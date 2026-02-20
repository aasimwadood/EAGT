"""
Early-Late Hybrid Fusion for Multimodal Affect Recognition.


This module implements:
- Early fusion: Concatenate modality embeddings
- Temporal modeling: BiLSTM for sequential dependencies
- Late fusion: Auxiliary per-modality classifiers with learnable weights
- Final prediction: Weighted combination per Equation (3)

Equation (3): ŷ_final = α·ŷ + β1·ŷ_face + β2·ŷ_speech + β3·ŷ_behav
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class FusionClassifier(nn.Module):
    """
    Basic early fusion classifier.

    For backward compatibility. Use FusionClassifierHybrid.
    """

    def __init__(
        self,
        face_dim: int = 512,
        audio_dim: int = 768,
        behav_dim: int = 16,
        hidden: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.input_dim = face_dim + audio_dim + behav_dim

        self.temporal = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, num_classes),
        )

    def forward(self, face: torch.Tensor, audio: torch.Tensor, behav: torch.Tensor) -> torch.Tensor:
        assert face.ndim == audio.ndim == behav.ndim == 3, "Inputs must be (B, T, D)"
        B, T, Df = face.shape
        Ba, Ta, Da = audio.shape
        Bb, Tb, Db = behav.shape
        assert B == Ba == Bb and T == Ta == Tb, "All modalities must share (B, T)"

        x = torch.cat([face, audio, behav], dim=-1)
        h, _ = self.temporal(x)
        h_pool = h.mean(dim=1)
        logits = self.head(h_pool)
        return logits


class FusionClassifierHybrid(nn.Module):
    """
    Early-Late Hybrid Fusion Classifier.


    This implements the full fusion strategy:
    1. Early fusion: Concatenate modality embeddings at each timestep
    2. BiLSTM: Encode temporal dependencies
    3. Late fusion: Auxiliary per-modality classifiers
    4. Weighted combination: ŷ_final = α·ŷ + β1·ŷ_face + β2·ŷ_speech + β3·ŷ_behav

    Parameters
    ----------
    face_dim : int
        Dimension of face embeddings (default 512 for ResNet18)
    audio_dim : int
        Dimension of audio embeddings (default 768 for wav2vec2)
    behav_dim : int
        Dimension of behavioral features (default 16)
    hidden : int
        Hidden size for BiLSTM
    num_layers : int
        Number of BiLSTM layers
    num_classes : int
        Number of affect classes (default 4: frustration, confusion, boredom, engagement)
    dropout : float
        Dropout probability
    learn_fusion_weights : bool
        If True, fusion weights are learnable parameters
    """

    def __init__(
        self,
        face_dim: int = 512,
        audio_dim: int = 768,
        behav_dim: int = 16,
        hidden: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.30,
        learn_fusion_weights: bool = True,
    ):
        super().__init__()

        self.face_dim = face_dim
        self.audio_dim = audio_dim
        self.behav_dim = behav_dim
        self.input_dim = face_dim + audio_dim + behav_dim
        self.hidden = hidden
        self.num_classes = num_classes

        # ============ Early Fusion: BiLSTM on concatenated features ============
        self.temporal = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Main fused classifier head
        self.fused_head = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, num_classes),
        )

        # ============ Late Fusion: Auxiliary per-modality classifiers ============

        # Face modality classifier
        self.face_encoder = nn.Sequential(
            nn.Linear(face_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.face_head = nn.Linear(hidden, num_classes)

        # Audio modality classifier
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.audio_head = nn.Linear(hidden, num_classes)

        # Behavioral modality classifier
        self.behav_encoder = nn.Sequential(
            nn.Linear(behav_dim, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.behav_head = nn.Linear(hidden // 2, num_classes)

        # ============ Learnable Fusion Weights (α, β1, β2, β3) ============
        # Initialized to give more weight to the fused prediction
        #  ŷ_final = α·ŷ + β1·ŷ_face + β2·ŷ_speech + β3·ŷ_behav
        if learn_fusion_weights:
            self.fusion_weights = nn.Parameter(
                torch.tensor([0.50, 0.20, 0.20, 0.10])  # [α, β1, β2, β3]
            )
        else:
            self.register_buffer(
                "fusion_weights",
                torch.tensor([0.50, 0.20, 0.20, 0.10])
            )

        # Optional: Attention-based fusion (alternative to fixed weights)
        self.use_attention_fusion = False
        if self.use_attention_fusion:
            self.fusion_attention = nn.Sequential(
                nn.Linear(4 * num_classes, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
                nn.Softmax(dim=-1),
            )

    def forward(
        self,
        face: torch.Tensor,
        audio: torch.Tensor,
        behav: torch.Tensor,
        return_all_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with hybrid fusion.

        Parameters
        ----------
        face : torch.Tensor
            Face embeddings of shape (B, T, Df)
        audio : torch.Tensor
            Audio embeddings of shape (B, T, Da)
        behav : torch.Tensor
            Behavioral features of shape (B, T, Db)
        return_all_logits : bool
            If True, return all intermediate logits for auxiliary losses

        Returns
        -------
        Dict[str, torch.Tensor]
            - "logits": Final fused logits (B, C)
            - "logits_fused": Main BiLSTM classifier logits
            - "logits_face": Face auxiliary classifier logits
            - "logits_audio": Audio auxiliary classifier logits
            - "logits_behav": Behavioral auxiliary classifier logits
            - "fusion_weights": Current fusion weights (4,)
        """
        # Validate input shapes
        assert face.ndim == audio.ndim == behav.ndim == 3, "Inputs must be (B, T, D)"
        B, T, Df = face.shape
        Ba, Ta, Da = audio.shape
        Bb, Tb, Db = behav.shape
        assert B == Ba == Bb, "Batch sizes must match"
        assert T == Ta == Tb, "Sequence lengths must match"

        # ============ Early Fusion Path ============
        # Concatenate modalities along feature dimension
        #  f_t = [f_t^face || f_t^speech || f_t^behav]
        x = torch.cat([face, audio, behav], dim=-1)  # (B, T, Df+Da+Db)

        # BiLSTM for temporal modeling
        #  h_t = BiLSTM(f_t, h_{t-1})
        h, _ = self.temporal(x)  # (B, T, 2*hidden)

        # Mean pooling over time (robust to variable lengths)
        h_pool = h.mean(dim=1)  # (B, 2*hidden)

        # Main fused classifier
        # ŷ_t = softmax(W·h_t + b)
        logits_fused = self.fused_head(h_pool)  # (B, C)

        # ============ Late Fusion Path: Auxiliary Classifiers ============
        # Pool unimodal features over time
        face_pool = face.mean(dim=1)   # (B, Df)
        audio_pool = audio.mean(dim=1)  # (B, Da)
        behav_pool = behav.mean(dim=1)  # (B, Db)

        # Per-modality predictions
        face_enc = self.face_encoder(face_pool)
        logits_face = self.face_head(face_enc)  # (B, C)

        audio_enc = self.audio_encoder(audio_pool)
        logits_audio = self.audio_head(audio_enc)  # (B, C)

        behav_enc = self.behav_encoder(behav_pool)
        logits_behav = self.behav_head(behav_enc)  # (B, C)

        # ============ Weighted Late Fusion ============
        #  ŷ_final = α·ŷ + β1·ŷ_face + β2·ŷ_speech + β3·ŷ_behav
        if self.use_attention_fusion:
            # Dynamic attention-based fusion
            all_logits = torch.cat([
                logits_fused, logits_face, logits_audio, logits_behav
            ], dim=-1)  # (B, 4*C)
            weights = self.fusion_attention(all_logits)  # (B, 4)
            weights = weights.unsqueeze(-1)  # (B, 4, 1)

            stacked = torch.stack([
                logits_fused, logits_face, logits_audio, logits_behav
            ], dim=1)  # (B, 4, C)
            logits_final = (weights * stacked).sum(dim=1)  # (B, C)
        else:
            # Fixed learnable weights
            weights = F.softmax(self.fusion_weights, dim=0)  # Normalize to sum to 1
            logits_final = (
                weights[0] * logits_fused +
                weights[1] * logits_face +
                weights[2] * logits_audio +
                weights[3] * logits_behav
            )

        output = {
            "logits": logits_final,
            "logits_fused": logits_fused,
            "logits_face": logits_face,
            "logits_audio": logits_audio,
            "logits_behav": logits_behav,
            "fusion_weights": F.softmax(self.fusion_weights, dim=0),
        }

        return output

    def get_fusion_weights(self) -> torch.Tensor:
        """Return normalized fusion weights."""
        return F.softmax(self.fusion_weights, dim=0)


class AuxiliaryLoss(nn.Module):
    """
    Compute combined loss with auxiliary classifier losses.

    This encourages each modality to learn discriminative features
    independently, improving robustness when modalities are missing.

    Parameters
    ----------
    main_weight : float
        Weight for main fused classifier loss
    aux_weight : float
        Weight for each auxiliary classifier loss
    """

    def __init__(
        self,
        main_weight: float = 1.0,
        aux_weight: float = 0.3,
        criterion: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.main_weight = main_weight
        self.aux_weight = aux_weight
        self.criterion = criterion or nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for all classifiers.

        Parameters
        ----------
        outputs : Dict
            Output from FusionClassifierHybrid.forward()
        targets : torch.Tensor
            Ground truth labels (B,)

        Returns
        -------
        Dict with 'total', 'main', 'face', 'audio', 'behav' losses
        """
        # Main losses
        loss_main = self.criterion(outputs["logits"], targets)
        loss_fused = self.criterion(outputs["logits_fused"], targets)

        # Auxiliary losses
        loss_face = self.criterion(outputs["logits_face"], targets)
        loss_audio = self.criterion(outputs["logits_audio"], targets)
        loss_behav = self.criterion(outputs["logits_behav"], targets)

        # Combined loss
        total_loss = (
            self.main_weight * loss_main +
            self.aux_weight * (loss_face + loss_audio + loss_behav)
        )

        return {
            "total": total_loss,
            "main": loss_main,
            "fused": loss_fused,
            "face": loss_face,
            "audio": loss_audio,
            "behav": loss_behav,
        }


def get_confidence_and_prediction(
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get prediction and confidence from logits.

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits of shape (B, C) or (C,)

    Returns
    -------
    Tuple of (predictions, confidences)
        predictions: Class indices
        confidences: Softmax probabilities of predicted class
    """
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = torch.max(probs, dim=-1)
    return predictions, confidences


class ModalityGating(nn.Module):
    """
    Learned gating mechanism to weight modalities based on quality/reliability.

    This can learn to down-weight noisy or missing modalities.
    """

    def __init__(
        self,
        face_dim: int = 512,
        audio_dim: int = 768,
        behav_dim: int = 16,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.face_gate = nn.Sequential(
            nn.Linear(face_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.audio_gate = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.behav_gate = nn.Sequential(
            nn.Linear(behav_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        face: torch.Tensor,
        audio: torch.Tensor,
        behav: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply learned gates to modalities.

        Returns gated features and gate values.
        """
        # Compute gates from pooled features
        face_pool = face.mean(dim=1) if face.ndim == 3 else face
        audio_pool = audio.mean(dim=1) if audio.ndim == 3 else audio
        behav_pool = behav.mean(dim=1) if behav.ndim == 3 else behav

        g_face = self.face_gate(face_pool)    # (B, 1)
        g_audio = self.audio_gate(audio_pool)  # (B, 1)
        g_behav = self.behav_gate(behav_pool)  # (B, 1)

        # Apply gates
        if face.ndim == 3:
            g_face = g_face.unsqueeze(1)
            g_audio = g_audio.unsqueeze(1)
            g_behav = g_behav.unsqueeze(1)

        gated_face = face * g_face
        gated_audio = audio * g_audio
        gated_behav = behav * g_behav

        gates = {
            "face": g_face.squeeze(-1),
            "audio": g_audio.squeeze(-1),
            "behav": g_behav.squeeze(-1),
        }

        return gated_face, gated_audio, gated_behav, gates
