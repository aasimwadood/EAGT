import torch
import torch.nn as nn

class FusionClassifier(nn.Module):
    """
    Early–late hybrid fusion for affect recognition.

    Inputs
    ------
    face  : (B, T, Df)  face/frame embeddings
    audio : (B, T, Da)  audio embeddings (e.g., Wav2Vec2 or MFCC-projected)
    behav : (B, T, Db)  behavioral features (interaction logs; engineered vectors)

    Processing
    ----------
    1) Concatenate modalities along feature dim at each timestep.
    2) BiLSTM encodes temporal dependencies.
    3) Mean-pool over time (robust to variable lengths / missing steps).
    4) Classification head → logits over 4 affect classes.

    Output
    ------
    logits : (B, C) with C = number of classes (default 4).
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
        """
        Parameters
        ----------
        face  : (B, T, Df)
        audio : (B, T, Da)
        behav : (B, T, Db)

        Returns
        -------
        logits : (B, C)
        """
        # Basic shape checks (helpful during integration)
        assert face.ndim == audio.ndim == behav.ndim == 3, "Inputs must be (B, T, D)"
        B, T, Df = face.shape
        Ba, Ta, Da = audio.shape
        Bb, Tb, Db = behav.shape
        assert B == Ba == Bb and T == Ta == Tb, "All modalities must share (B, T)"

        x = torch.cat([face, audio, behav], dim=-1)  # (B, T, Df+Da+Db)

        h, _ = self.temporal(x)        # (B, T, 2H)
        h_pool = h.mean(dim=1)         # (B, 2H) mean over time (robust to variable-length windows)
        logits = self.head(h_pool)     # (B, C)
        return logits
