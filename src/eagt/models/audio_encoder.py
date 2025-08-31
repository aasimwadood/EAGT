import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class AudioEncoder(nn.Module):
    """
    Audio encoder using HuggingFace Wav2Vec2.

    Returns contextualized embeddings for each timestep.
    """
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.feat_dim = self.model.config.hidden_size

    def forward(self, wav: torch.Tensor):
        """
        Parameters
        ----------
        wav : torch.Tensor, shape (B, T)
            Audio waveform batches.
        
        Returns
        -------
        feats : torch.Tensor, shape (B, T_frames, D)
        """
        out = self.model(wav).last_hidden_state
        return out
